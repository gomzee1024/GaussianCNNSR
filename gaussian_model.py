import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Helper Function: Get Pixel Coordinates ---
# We cache this grid to avoid re-computing it every time
_pixel_grid_cache = {}


def get_pixel_coordinates(h, w, device):
    """
    Get a (h, w, 2) grid of (x, y) coordinates in the range [-1, 1].
    """
    key = (h, w, str(device))
    if key not in _pixel_grid_cache:
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing='ij'
        )
        grid = torch.stack([x_coords, y_coords], dim=-1)  # [h, w, 2]
        _pixel_grid_cache[key] = grid
    return _pixel_grid_cache[key]


# --- Main Gaussian Model ---

class GaussianModel(nn.Module):
    """
    A model that takes a low-res image and predicts parameters
    for a 2D Gaussian field.
    """

    def __init__(self, in_channels=3, n_gaussians_per_pixel=2):
        super().__init__()
        # --- FIX: Each Gaussian now has 9 params ---
        # 2 for position offset (dx, dy)
        # 3 for covariance (v1_x, v2_x, v2_y)
        # 3 for color (r, g, b)
        # 1 for opacity (a)
        # Total = 9
        n_params = 9 * n_gaussians_per_pixel

        # Simple CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            # Output 9 parameters per pixel
            nn.Conv2d(128, n_params, 1, 1, 0)
        )

    def forward(self, lr_image):
        """
        Input: lr_image (B, 3, h_in, w_in)
        Output: params (B, 9, h_in, w_in)
        """
        params = self.encoder(lr_image)
        return params


class GaussianRenderer(nn.Module):
    """
    Renders a Gaussian parameter map at a given resolution.
    This module has no learnable parameters.
    """

    def __init__(self):
        super().__init__()

    def create_base_grid(self, B, h_in, w_in, device):
        """Creates a [B, h_in, w_in, 2] grid of base (x, y) coordinates."""
        # This is the grid of the *centers* of the LR pixels
        y, x = torch.meshgrid(
            torch.linspace(-1 + 1 / h_in, 1 - 1 / h_in, h_in, device=device),
            torch.linspace(-1 + 1 / w_in, 1 - 1 / w_in, w_in, device=device),
            indexing='ij'
        )
        grid = torch.stack([x, y], dim=-1)  # [h_in, w_in, 2]
        return grid.expand(B, -1, -1, -1)  # [B, h_in, w_in, 2]

    def build_positions(self, params, h_in, w_in):
        """Builds mean (mu) vectors. [B, h_in, w_in, 2]"""
        # Params[0:2] are position offsets (dx, dy)
        pos_offsets = params[:, 0:2, :, :]  # [B, 2, h_in, w_in]
        pos_offsets = pos_offsets.permute(0, 2, 3, 1)  # [B, h_in, w_in, 2]

        B = params.shape[0]
        base_grid = self.create_base_grid(B, h_in, w_in, device=params.device)

        # Apply offset, scaled to be a fraction of a pixel
        # Use tanh to keep offsets bounded
        scale = (2.0 / h_in)  # Scale offset to LR pixel size
        mu = base_grid + torch.tanh(pos_offsets) * scale
        return mu

    def build_covariance(self, params):
        """Builds covariance matrices. [B, h_in, w_in, 2, 2]"""
        # Params[2:5] are covariance parameters
        cov_params = params[:, 2:5, :, :]  # [B, 3, h_in, w_in]
        cov_params = cov_params.permute(0, 2, 3, 1)  # [B, h_in, w_in, 3]

        # Cholesky decomposition: L = [[v1_x, 0], [v2_x, v2_y]]
        # This ensures the matrix is positive semi-definite
        v1_x = torch.exp(cov_params[..., 0])  # Ensure > 0
        v2_x = cov_params[..., 1]  # Can be any value
        v2_y = torch.exp(cov_params[..., 2])  # Ensure > 0

        # Create L
        v1_y = torch.zeros_like(v1_x)
        row1 = torch.stack([v1_x, v1_y], dim=-1)  # [B, h_in, w_in, 2]
        row2 = torch.stack([v2_x, v2_y], dim=-1)  # [B, h_in, w_in, 2]

        L = torch.stack([row1, row2], dim=-2)  # [B, h_in, w_in, 2, 2]

        # Covariance Sigma = L * L^T
        cov_matrices = L @ L.transpose(-1, -2)
        return cov_matrices

    def build_colors(self, params):
        """Builds color vectors. [B, h_in, w_in, 3]"""
        # Params[5:8] are color (r, g, b)
        color_params = params[:, 5:8, :, :]  # [B, 3, h_in, w_in]
        colors = color_params.permute(0, 2, 3, 1)  # [B, h_in, w_in, 3]
        # Use sigmoid to keep colors in [0, 1] range
        return torch.sigmoid(colors)

    # --- NEW FUNCTION ---
    def build_opacity(self, params):
        """Builds opacity values. [B, h_in, w_in, 1]"""
        # Params[8:9] is opacity (a)
        opacity_params = params[:, 8:9, :, :]  # [B, 1, h_in, w_in]
        opacities = opacity_params.permute(0, 2, 3, 1)  # [B, h_in, w_in, 1]
        # Use sigmoid to keep opacity in [0, 1] range
        return torch.sigmoid(opacities)

    # --- THIS IS THE FIX ---
    def forward(self, params, h_out, w_out):
        """
        Renders the Gaussian parameters.
        Args:
            params (torch.Tensor): [B, 9, h_in, w_in] tensor from the model.
            h_out (int): The output height (e.g., 256).
            w_out (int): The output width (e.g., 256).
        """
        # h_out and w_out are now passed directly
        hr_coords = get_pixel_coordinates(h_out, w_out, device=params.device)
        # --- END OF FIX ---

        B, _, h_in, w_in = params.shape

        # --- 1. Build Gaussian components ---
        cov = self.build_covariance(params)  # [B, h_in, w_in, 2, 2]
        mu = self.build_positions(params, h_in, w_in)  # [B, h_in, w_in, 2]
        colors = self.build_colors(params)  # [B, h_in, w_in, 3]
        opacities = self.build_opacity(params)  # [B, h_in, w_in, 1]

        # --- 2. Reshape for broadcasting ---
        # We want to compare every HR coord with every Gaussian
        N = h_in * w_in  # Total number of Gaussians

        mu = mu.view(B, N, 2)  # [B, N, 2]
        cov = cov.view(B, N, 2, 2)  # [B, N, 2, 2]
        colors = colors.view(B, N, 3)  # [B, N, 3]
        opacities = opacities.view(B, N, 1)  # [B, N, 1]

        # Add a small value for numerical stability before inverting
        cov = cov + torch.eye(2, device=params.device) * 1e-5
        Sigma_inv = torch.inverse(cov)  # [B, N, 2, 2]

        # --- 3. Expand tensors for splatting ---
        # Target shape for splatting: [B, N, h_out, w_out, ...]

        hr_coords_exp = hr_coords.view(1, 1, h_out, w_out, 2)
        mu_exp = mu.view(B, N, 1, 1, 2).expand(-1, -1, h_out, w_out, -1)
        colors_exp = colors.view(B, N, 1, 1, 3).expand(-1, -1, h_out, w_out, -1)
        Sigma_inv_exp = Sigma_inv.view(B, N, 1, 1, 2, 2).expand(-1, -1, h_out, w_out, -1, -1)
        opacities_exp = opacities.view(B, N, 1, 1, 1).expand(-1, -1, h_out, w_out, -1)

        # --- 4. Calculate Gaussian weights ---
        # This is the core splatting math: w = alpha * exp(-0.5 * (x-mu)^T * Sigma_inv * (x-mu))

        dx = hr_coords_exp - mu_exp  # [B, N, h_out, w_out, 2]

        # Reshape for matrix multiplication
        dx_T = dx.unsqueeze(-2)  # [B, N, h_out, w_out, 1, 2]
        dx = dx.unsqueeze(-1)  # [B, N, h_out, w_out, 2, 1]

        # (dx_T @ Sigma_inv @ dx)
        exponent = -0.5 * (dx_T @ Sigma_inv_exp @ dx)
        exponent = exponent.squeeze(-1).squeeze(-1)  # [B, N, h_out, w_out]

        # Prevent numerical overflow/underflow
        exponent = torch.clamp(exponent, min=-20, max=20)
        gaussian_kernel = torch.exp(exponent)  # [B, N, h_out, w_out]

        # --- FIX: Modulate kernel by opacity ---
        weights = opacities_exp.squeeze(-1) * gaussian_kernel  # [B, N, h_out, w_out]

        # --- 5. Render final image ---
        # (Accumulated Blending)

        # Add a small epsilon to weights to avoid division by zero
        weights_sum = weights.sum(dim=1, keepdim=True) + 1e-8  # [B, 1, h_out, w_out]

        # Normalize weights
        normalized_weights = (weights / weights_sum).unsqueeze(-1)  # [B, N, h_out, w_out, 1]

        # Weighted sum of colors
        # (B, N, h, w, 1) * (B, N, h, w, 3) -> (B, N, h, w, 3)
        weighted_colors = normalized_weights * colors_exp

        # Sum across all Gaussians
        image = weighted_colors.sum(dim=1)  # [B, h_out, w_out, 3]

        # Permute to (B, C, H, W) for PyTorch convention
        image = image.permute(0, 3, 1, 2)  # [B, 3, h_out, w_out]

        return image

