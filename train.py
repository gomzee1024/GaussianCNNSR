import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import argparse
import os
from tqdm import tqdm

from gaussian_model import GaussianModel, GaussianRenderer
from dataset import CustomImageDataset


def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Create Dataset and DataLoader ---
    try:
        train_dataset = CustomImageDataset(
            root_dir=args.dataset_path,
            hr_size=args.hr_size,
            lr_scale=args.scale
        )
    except FileNotFoundError:
        print("\n---!! FATAL ERROR !!----")
        print(f"Could not find the dataset directory: {args.dataset_path}")
        print("Please download the DIV2K dataset and place the HR images")
        print("in a folder (e.g., 'DIV2K_train_HR') and point --dataset_path to it.")
        print("--------------------------")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    print(f"Loaded dataset with {len(train_dataset)} images.")

    # --- 2. Initialize Model, Renderer, Optimizer, Loss ---
    model = GaussianModel().to(device)
    # Renderer is not learnable, just a utility class
    renderer = GaussianRenderer()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.L1Loss()  # L1 loss is common for sharp images

    # --- 3. Training Loop ---
    model.train()

    for epoch in range(args.epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        total_loss = 0

        for i, (lr_images, hr_images) in enumerate(progress_bar):
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)

            # --- Forward Pass ---
            optimizer.zero_grad()

            # Get Gaussian parameters from the model
            gaussians_params = model(lr_images)

            # Render the HR image from the Gaussians
            rendered_images = renderer(
                gaussians_params,
                h_out=args.hr_size,
                w_out=args.hr_size
            )

            # --- Backward Pass ---
            loss = criterion(rendered_images, hr_images)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{total_loss / (i + 1):.6f}")

            # Save a sample image
            if i == 0 and (epoch + 1) % 5 == 0:
                sample_output = torch.cat([hr_images[:4], rendered_images[:4]], dim=0)
                os.makedirs('train_samples', exist_ok=True)
                save_image(
                    sample_output,
                    f"train_samples/epoch_{epoch + 1}.png",
                    nrow=4,
                    normalize=True
                )

    # --- 4. Save the trained model ---
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"\nTraining complete. Model saved to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Gaussian SR Model")

    parser.add_argument('--dataset_path', type=str, required=True,
                        help="Path to the directory containing HR training images (e.g., DIV2K_train_HR)")
    parser.add_argument('--save_path', type=str, default='models/gaussian_sr.pth',
                        help="Path to save the trained model weights")
    parser.add_argument('--scale', type=int, default=4,
                        help="Downscaling factor for LR images (e.g., 4 for 4x SR)")
    parser.add_argument('--hr_size', type=int, default=256,
                        help="Crop size for HR images")
    parser.add_argument('--epochs', type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Training batch size")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate")

    args = parser.parse_args()
    main(args)
