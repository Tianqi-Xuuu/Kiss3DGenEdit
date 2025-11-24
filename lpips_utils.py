"""
LPIPS (Learned Perceptual Image Patch Similarity) utility functions
for computing perceptual distance between images.
"""

import lpips
import torch
from PIL import Image
import torchvision.transforms as transforms
from typing import Union, Tuple, Optional


class LPIPSCalculator:
    """Class to calculate LPIPS distance between images."""

    def __init__(self, net: str = 'alex', device: Optional[str] = None):
        """
        Initialize LPIPS calculator.

        Args:
            net: Network to use ('alex' or 'vgg'). 'alex' gives best forward scores,
                 'vgg' is closer to traditional perceptual loss.
            device: Device to run on ('cuda' or 'cpu'). If None, auto-detect.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.loss_fn = lpips.LPIPS(net=net).to(self.device)
        self.loss_fn.eval()

        # Define image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Resize images
            transforms.ToTensor(),         # Convert to tensor, range [0,1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1,1]
        ])

    def calculate_distance(self,
                          img1: Union[str, Image.Image, torch.Tensor],
                          img2: Union[str, Image.Image, torch.Tensor]) -> float:
        """
        Calculate LPIPS distance between two images.

        Args:
            img1: First image (path, PIL Image, or torch Tensor)
            img2: Second image (path, PIL Image, or torch Tensor)

        Returns:
            LPIPS distance as float
        """
        # Load and preprocess images
        img1_tensor = self._prepare_image(img1)
        img2_tensor = self._prepare_image(img2)

        # Calculate LPIPS distance
        with torch.no_grad():
            distance = self.loss_fn(img1_tensor, img2_tensor)

        return distance.item()

    def _prepare_image(self, img: Union[str, Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Prepare image for LPIPS calculation.

        Args:
            img: Image as path string, PIL Image, or torch Tensor

        Returns:
            Preprocessed image tensor with batch dimension
        """
        if isinstance(img, str):
            # Load from path
            pil_img = Image.open(img).convert('RGB')
            tensor = self.transform(pil_img)
        elif isinstance(img, Image.Image):
            # PIL Image
            pil_img = img.convert('RGB')
            tensor = self.transform(pil_img)
        elif isinstance(img, torch.Tensor):
            # Already a tensor
            if img.dim() == 3:
                tensor = img
            elif img.dim() == 4:
                # If batch dimension exists, take first image
                tensor = img[0]
            else:
                raise ValueError(f"Unexpected tensor dimensions: {img.dim()}")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

        # Add batch dimension if needed
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)


def calculate_lpips_distance(img1_path: str,
                            img2_path: str,
                            net: str = 'alex') -> Tuple[float, Optional[float]]:
    """
    Convenience function to calculate LPIPS distance between two image files.

    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        net: Network to use ('alex', 'vgg', or 'both')

    Returns:
        If net='both': tuple of (alex_distance, vgg_distance)
        Otherwise: tuple of (distance, None)
    """
    if net == 'both':
        # Calculate with both networks
        calc_alex = LPIPSCalculator(net='alex')
        calc_vgg = LPIPSCalculator(net='vgg')

        alex_dist = calc_alex.calculate_distance(img1_path, img2_path)
        vgg_dist = calc_vgg.calculate_distance(img1_path, img2_path)

        return alex_dist, vgg_dist
    else:
        # Calculate with single network
        calc = LPIPSCalculator(net=net)
        dist = calc.calculate_distance(img1_path, img2_path)
        return dist, None


def batch_calculate_lpips(reference_img: Union[str, Image.Image],
                          comparison_imgs: list,
                          net: str = 'alex') -> list:
    """
    Calculate LPIPS distance from one reference image to multiple comparison images.

    Args:
        reference_img: Reference image (path or PIL Image)
        comparison_imgs: List of comparison images (paths or PIL Images)
        net: Network to use ('alex' or 'vgg')

    Returns:
        List of LPIPS distances
    """
    calc = LPIPSCalculator(net=net)
    distances = []

    for comp_img in comparison_imgs:
        dist = calc.calculate_distance(reference_img, comp_img)
        distances.append(dist)

    return distances


if __name__ == "__main__":
    # Example usage
    import os

    # Test with example images if they exist
    img1_path = "examples/chair_armed.png"
    img2_path = "examples/chair_comfort.jpg"

    if os.path.exists(img1_path) and os.path.exists(img2_path):
        print("Testing LPIPS calculation...")

        # Method 1: Using convenience function
        alex_dist, vgg_dist = calculate_lpips_distance(img1_path, img2_path, net='both')
        print(f"\nMethod 1 - Convenience function:")
        print(f"LPIPS distance (Alex): {alex_dist:.4f}")
        print(f"LPIPS distance (VGG): {vgg_dist:.4f}")

        # Method 2: Using class
        calc = LPIPSCalculator(net='alex')
        dist = calc.calculate_distance(img1_path, img2_path)
        print(f"\nMethod 2 - Class method:")
        print(f"LPIPS distance (Alex): {dist:.4f}")
    else:
        print(f"Example images not found at {img1_path} and {img2_path}")
        print("Please update the paths to test the functions.")