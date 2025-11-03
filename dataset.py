import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path


class CustomImageDataset(Dataset):
    """
    A custom PyTorch Dataset to load images from a directory.
    It returns a high-resolution (HR) image and a corresponding
    low-resolution (LR) image created by downsampling.
    """

    def __init__(self, root_dir, hr_size=256, lr_scale=4):
        """
        Args:
            root_dir (str): Directory with all the images.
            hr_size (int): The size to crop HR images to.
            lr_scale (int): The factor to downscale HR to get LR.
        """
        self.root_dir = Path(root_dir)
        self.image_paths = sorted(list(self.root_dir.glob('*.png')))
        self.hr_size = hr_size
        self.lr_size = hr_size // lr_scale

        # --- FIX: Define transforms for PIL Images only ---

        # 1. Transform to get the cropped HR PIL image
        self.hr_crop_transform = transforms.RandomCrop(hr_size)

        # 2. Transform to get the LR PIL image from the cropped HR
        self.lr_resize_transform = transforms.Resize(
            self.lr_size,
            interpolation=transforms.InterpolationMode.BICUBIC
        )

        # 3. Transform to convert a PIL image to a Tensor
        self.tensor_transform = transforms.ToTensor()
        # --- End of Fix ---

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image or skip
            return self.__getitem__((idx + 1) % len(self))

        # --- FIX: Apply transforms in the correct order ---
        # This is the section that was corrected.

        # 1. Create the cropped HR PIL image
        hr_pil_cropped = self.hr_crop_transform(image)

        # 2. Create the LR PIL image *from the cropped HR PIL image*
        lr_pil = self.lr_resize_transform(hr_pil_cropped)

        # 3. Convert both final PIL images to Tensors
        hr_image = self.tensor_transform(hr_pil_cropped)
        lr_image = self.tensor_transform(lr_pil)
        # --- End of Fix ---

        return lr_image, hr_image


if __name__ == '__main__':
    # --- Test the dataset loader ---
    # This assumes you have a folder named 'DIV2K_train_HR' in your current directory
    # If your path is different, change it here.
    try:
        dataset = CustomImageDataset(root_dir='./DIV2K_train_HR')
        print(f"Found {len(dataset)} images.")

        # Get one sample
        lr_img, hr_img = dataset[0]

        print(f"LR image shape: {lr_img.shape}")  # Should be [3, 64, 64]
        print(f"HR image shape: {hr_img.shape}")  # Should be [3, 256, 256]
        print("Dataset test successful!")

    except FileNotFoundError:
        print("\n---!! ERROR !!----")
        print("Could not find the 'DIV2K_train_HR' directory.")
        print("Please download the DIV2K dataset and place the HR images")
        print("in a folder named 'DIV2K_train_HR' in this directory.")
        print("--------------------")
    except Exception as e:
        print(f"An error occurred: {e}")