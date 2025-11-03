import torch
import argparse
import os
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from gaussian_model import GaussianModel, GaussianRenderer


def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Model ---
    model = GaussianModel().to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        print("Please train the model first using 'train.py'")
        return

    model.eval()

    # Renderer is not learnable
    renderer = GaussianRenderer()

    # --- 2. Load Input Image ---
    try:
        input_image = Image.open(args.input_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Input image not found at {args.input_path}")
        return

    # Prepare image
    lr_transform = transforms.ToTensor()
    lr_image = lr_transform(input_image).unsqueeze(0).to(device)

    _, _, h_lr, w_lr = lr_image.shape
    h_out = h_lr * args.scale
    w_out = w_lr * args.scale

    print(f"Loading model from {args.model_path}")
    print(f"Rendering {args.input_path} ({w_lr}x{h_lr}) at {args.scale}x scale -> ({w_out}x{h_out})")

    # --- 3. Run Inference and Render ---
    with torch.no_grad():
        # Get Gaussian parameters from the model
        gaussians_params = model(lr_image)

        # Render the HR image
        rendered_image = renderer(
            gaussians_params,
            h_out=h_out,
            w_out=w_out
        )

    # --- 4. Save Output Image ---
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    save_image(rendered_image.squeeze(0), args.output_path)

    print(f"\nDone. Super-resolved image saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a super-resolved image using a Gaussian SR Model")

    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the trained model weights (e.g., models/gaussian_sr.pth)")
    parser.add_argument('--input_path', type=str, required=True,
                        help="Path to the low-resolution input image")
    parser.add_argument('--output_path', type=str, default='output/rendered_image.png',
                        help="Path to save the super-resolved output image")
    parser.add_argument('--scale', type=int, default=4,
                        help="Upscaling factor (e.g., 4 for 4x SR). Must match training scale.")

    args = parser.parse_args()
    main(args)
