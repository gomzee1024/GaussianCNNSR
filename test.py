import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path
import argparse
from tqdm import tqdm

# We'll use torchmetrics for standard PSNR and SSIM calculations
# Install it with: pip install torchmetrics
try:
    from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
except ImportError:
    print("Error: 'torchmetrics' library not found.")
    print("Please install it by running: pip install torchmetrics")
    exit()

# Import our models from the existing file
from gaussian_model import GaussianModel, GaussianRenderer


class ValidationDataset(Dataset):
    """
    A dataset for loading validation images.
    It performs a CENTER CROP to get a consistent patch,
    then creates the LR image from that crop.
    """

    def __init__(self, root_dir, hr_size=256, lr_scale=4):
        """
        Args:
            root_dir (str): Directory with all the images.
            hr_size (int): The size to crop HR images to. Must match training.
            lr_scale (int): The downscaling factor. Must match training.
        """
        self.root_dir = Path(root_dir)
        self.image_paths = sorted(list(self.root_dir.glob('*.png')))

        # 1. Transform to get a consistent HR crop
        self.hr_transform = transforms.Compose([
            transforms.CenterCrop(hr_size)
        ])

        # 2. Transform to create the LR version from the HR crop
        self.lr_transform = transforms.Compose([
            transforms.Resize(
                (hr_size // lr_scale, hr_size // lr_scale),
                interpolation=transforms.InterpolationMode.BICUBIC
            )
        ])

        # 3. Transform to convert PIL images to Tensors
        self.tensor_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        # 1. Create the cropped HR PIL image
        hr_pil_cropped = self.hr_transform(image)

        # 2. Create the LR PIL image *from the cropped HR PIL image*
        lr_pil = self.lr_transform(hr_pil_cropped)

        # 3. Convert both to Tensors
        hr_image = self.tensor_transform(hr_pil_cropped)
        lr_image = self.tensor_transform(lr_pil)

        return lr_image, hr_image


def create_comparison_image(hr_pil, rendered_pil, psnr, ssim, text_height=40, border=10):
    """
    Combines two PIL images side-by-side with metrics written below.
    """
    width, height = hr_pil.size

    # Create a new white canvas
    total_width = width * 2 + border * 3
    total_height = height + text_height + border

    canvas = Image.new('RGB', (total_width, total_height), 'white')
    draw = ImageDraw.Draw(canvas)

    # Use the default PIL font (requires no .ttf file)
    try:
        font = ImageFont.load_default()
    except IOError:
        print("Cannot load default font. Using a basic font.")
        font = None

    # Paste the images
    canvas.paste(hr_pil, (border, border))
    canvas.paste(rendered_pil, (width + border * 2, border))

    # Draw the text
    text_y = height + border + 5
    draw.text((border, text_y), "Ground Truth (Center Crop)", fill="black", font=font)

    metrics_text = f"Rendered Output | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}"
    draw.text((width + border * 2, text_y), metrics_text, fill="black", font=font)

    return canvas


def main(args):
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    if not os.path.exists(args.data_path):
        print(f"Error: Validation data directory not found at {args.data_path}")
        return

    # --- NEW: Create all output directories ---
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.gt_dir, exist_ok=True)
    os.makedirs(args.ren_dir, exist_ok=True)
    print(f"Saving comparison images to: {args.output_dir}")
    print(f"Saving Ground Truth images to: {args.gt_dir}")
    print(f"Saving Rendered images to: {args.ren_dir}")

    # --- 2. Load Model ---
    checkpoint = torch.load(args.model_path, map_location=device)

    model = GaussianModel().to(device)
    renderer = GaussianRenderer().to(device)

    # --- THIS IS THE FIX ---
    # We will try three common ways to load the weights:
    try:
        # 1. Try loading from a dictionary with key 'model_state_dict' (from train.py)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded weights using key: 'model_state_dict' (from train.py)")
    except KeyError:
        # 2. Try loading from a dictionary with key 'state_dict' (from gaussify.py)
        try:
            model.load_state_dict(checkpoint['state_dict'])
            print("Loaded weights using key: 'state_dict' (from gaussify.py)")
        except KeyError:
            # 3. Try loading the checkpoint directly (assuming it's just the state_dict)
            try:
                model.load_state_dict(checkpoint)
                print("Loaded weights directly from the file (legacy format).")
            except Exception as e:
                # If all three fail, then we give up.
                print("Error: Could not find model weights in the checkpoint file.")
                print("The .pth file must be a state_dict or a dictionary")
                print("containing 'model_state_dict' or 'state_dict'.")
                print(f"Internal error: {e}")
                return
    except Exception as e:
        # This catches other errors, like mismatched keys in the state_dict
        print(f"Error loading state_dict. Model architecture may have changed: {e}")
        return
    # --- END OF FIX ---

    model.eval()
    renderer.eval()
    print(f"Successfully loaded trained model from {args.model_path}")

    # --- 3. Load Data ---
    dataset = ValidationDataset(
        root_dir=args.data_path,
        hr_size=args.hr_size,
        lr_scale=args.lr_scale
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    print(f"Found {len(dataset)} validation images in {args.data_path}")

    # --- 4. Setup Metrics ---
    # Metrics for the final average
    avg_psnr_metric = PeakSignalNoiseRatio().to(device)
    avg_ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # Metrics for per-image calculation (re-created each time)
    psnr_per_image = PeakSignalNoiseRatio().to(device)
    ssim_per_image = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # --- 5. Run Validation Loop ---
    progress_bar = tqdm(dataloader, desc="Validating")

    for i, (lr_images, hr_images) in enumerate(progress_bar):
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)

        with torch.no_grad():
            params = model(lr_images)
            rendered_images = renderer(
                params,
                h_out=args.hr_size,
                w_out=args.hr_size
            )

        # Update the overall average metrics
        avg_psnr_metric.update(rendered_images, hr_images)
        avg_ssim_metric.update(rendered_images, hr_images)

        # --- MODIFIED: Save all 3 versions of images ---
        for j in range(rendered_images.shape[0]):
            hr_image_single = hr_images[j]
            rendered_image_single = rendered_images[j]

            # Calculate per-image metrics
            psnr_val = psnr_per_image(rendered_image_single[None], hr_image_single[None]).item()
            ssim_val = ssim_per_image(rendered_image_single[None], hr_image_single[None]).item()

            # Get original filename
            image_index = i * args.batch_size + j
            base_filename = dataset.image_paths[image_index].name

            # Convert tensors to PIL images for saving
            hr_pil = TF.to_pil_image(hr_image_single.cpu())
            rendered_pil = TF.to_pil_image(rendered_image_single.cpu())

            # 1. Save Ground Truth image
            hr_save_path = os.path.join(args.gt_dir, base_filename)
            hr_pil.save(hr_save_path)

            # 2. Save Rendered image
            ren_save_path = os.path.join(args.ren_dir, base_filename)
            rendered_pil.save(ren_save_path)

            # 3. Create and save the side-by-side comparison
            comparison_image = create_comparison_image(hr_pil, rendered_pil, psnr_val, ssim_val)
            comp_save_path = os.path.join(args.output_dir, base_filename)
            comparison_image.save(comp_save_path)

    # --- 6. Compute Final Scores ---
    avg_psnr = avg_psnr_metric.compute()
    avg_ssim = avg_ssim_metric.compute()

    print("\n--- Validation Complete ---")
    print(f"Dataset: {args.data_path}")
    print(f"Model:   {args.model_path}")
    print(f"Comparison Output:  {args.output_dir}")
    print(f"GT Output:          {args.gt_dir}")
    print(f"Rendered Output:    {args.ren_dir}")
    print(f"Average PSNR: {avg_psnr.item():.4f} dB")
    print(f"Average SSIM: {avg_ssim.item():.4f}")
    print("---------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run validation on a trained GaussianModel."
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="Path to the trained model .pth file (e.g., 'models/my_model.pth')."
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help="Path to the validation HR images folder (e.g., './DIV2K_valid_HR')."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="./validation_output",
        help="Directory to save comparison images."
    )
    # --- NEW ARGUMENTS ---
    parser.add_argument(
        '--gt_dir',
        type=str,
        default="./validationGT",
        help="Directory to save the Ground Truth images."
    )
    parser.add_argument(
        '--ren_dir',
        type=str,
        default="./validationRen",
        help="Directory to save the Rendered images."
    )
    # --- END NEW ARGUMENTS ---
    parser.add_argument(
        '--hr_size',
        type=int,
        default=256,
        help="HR crop size. Must match the 'hr_size' used during training."
    )
    parser.add_argument(
        '--lr_scale',
        type=int,
        default=4,
        help="Downscaling factor. Must match the 'lr_scale' used during training."
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help="Batch size for validation. Use 1 if you have low VRAM."
    )

    args = parser.parse_args()
    main(args)

