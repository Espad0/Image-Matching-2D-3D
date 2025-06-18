"""Base class for feature matchers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from pathlib import Path


class BaseMatcher(ABC):
    """
    Abstract base class for feature matchers.
    
    This class defines the interface that all feature matchers must implement,
    ensuring consistency across different matching algorithms.
    """
    
    def __init__(self, device: torch.device, config: Dict):
        """
        Initialize the matcher.
        
        Args:
            device: PyTorch device for computation
            config: Configuration dictionary
        """
        self.device = device
        self.config = config
        self._setup()
    
    @abstractmethod
    def _setup(self):
        """Setup the matcher model and parameters."""
        pass
    
    @abstractmethod
    def match_pair(
        self,
        image1_path: str,
        image2_path: str,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match features between two images.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            **kwargs: Additional arguments
        
        Returns:
            Tuple of:
                - keypoints1: Nx2 array of keypoints in image1
                - keypoints2: Nx2 array of keypoints in image2
                - confidence: Nx1 array of match confidences
        """
        pass
    
    def match_pairs(
        self,
        image_paths: List[str],
        pairs: List[Tuple[int, int]],
        output_dir: Path,
        verbose: bool = True
    ) -> Dict:
        """
        Match features for multiple image pairs.
        
        Args:
            image_paths: List of all image paths
            pairs: List of (idx1, idx2) pairs to match
            output_dir: Directory to save matches
            verbose: Show progress bar
        
        Returns:
            Dictionary of matches
        """
        from tqdm import tqdm
        
        matches = {}
        iterator = tqdm(pairs, desc="Matching image pairs") if verbose else pairs
        
        for idx1, idx2 in iterator:
            img1_path = image_paths[idx1]
            img2_path = image_paths[idx2]
            
            # Get matches
            kpts1, kpts2, conf = self.match_pair(img1_path, img2_path)
            
            # Store matches
            key = (idx1, idx2)
            matches[key] = {
                'keypoints1': kpts1,
                'keypoints2': kpts2,
                'confidence': conf,
                'num_matches': len(kpts1)
            }
        
        return matches
    
    def apply_tta(
        self,
        image: torch.Tensor,
        tta_type: str
    ) -> Tuple[torch.Tensor, callable]:
        """
        Apply test-time augmentation to image.
        
        Args:
            image: Input image tensor
            tta_type: Type of augmentation
        
        Returns:
            Augmented image and inverse transformation function
        """
        if tta_type == 'orig':
            return image, lambda x: x
        elif tta_type == 'flip_lr':
            return torch.flip(image, [3]), lambda x: self._flip_lr_points(x, image.shape[3])
        elif tta_type == 'flip_ud':
            return torch.flip(image, [2]), lambda x: self._flip_ud_points(x, image.shape[2])
        else:
            raise ValueError(f"Unknown TTA type: {tta_type}")
    
    def _flip_lr_points(self, points: np.ndarray, width: int) -> np.ndarray:
        """Flip points horizontally."""
        points = points.copy()
        points[:, 0] = width - points[:, 0] - 1
        return points
    
    def _flip_ud_points(self, points: np.ndarray, height: int) -> np.ndarray:
        """Flip points vertically."""
        points = points.copy()
        points[:, 1] = height - points[:, 1] - 1
        return points