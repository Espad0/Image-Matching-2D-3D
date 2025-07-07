"""
Image pair selection using global descriptors.

This module solves a critical problem in 3D reconstruction: which images should
we try to match? With N images, there are N*(N-1)/2 possible pairs. For 100 
images, that's 4,950 pairs! We need to be smarter.

The solution: Use AI to create a "fingerprint" (global descriptor) for each image,
then select pairs with similar fingerprints.

How it works:
1. Pass each image through a pre-trained CNN (like EfficientNet)
2. Extract a feature vector that captures the image's content
3. Compare feature vectors to find similar images
4. Select pairs that are similar but not identical

Why this matters:
- 100 images: 4,950 possible pairs → ~500 selected pairs (10x speedup!)
- 1000 images: 499,500 possible pairs → ~5,000 selected pairs (100x speedup!)

The key insight: Images that will match well usually have similar content
(same building, same scene), so we can predict which pairs are worth trying.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import cv2
from tqdm import tqdm
import logging
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

logger = logging.getLogger(__name__)


def get_unique_indices(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Get indices of first occurrence of unique elements in tensor."""
    unique, idx, counts = torch.unique(
        tensor, dim=dim, sorted=True, 
        return_inverse=True, return_counts=True
    )
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0], device=cum_sum.device), cum_sum[:-1]))
    first_indices = ind_sorted[cum_sum]
    return first_indices


def get_image_pairs_exhaustive(img_filenames: List[str]) -> List[Tuple[int, int]]:
    """Generate all possible image pairs for exhaustive matching."""
    pairs = []
    n = len(img_filenames)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    return pairs


def compute_global_descriptors(filenames: List[str], 
                             model_name: str = None,
                             model_path: str = None,
                             device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Compute global descriptors for images using a pretrained CNN.
    
    Args:
        filenames: List of image file paths
        model_name: Name of the timm model to use
        model_path: Path to model weights
        device: Torch device
        
    Returns:
        Global descriptors tensor
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = model_name or 'tf_efficientnet_b7'
    
    # Create model
    if model_path and os.path.exists(model_path):
        model = timm.create_model(model_name, checkpoint_path=model_path)
        logger.info(f"Loaded model from {model_path}")
    else:
        model = timm.create_model(model_name, pretrained=True)
        logger.info(f"Using pretrained {model_name}")
    
    model = model.eval().to(device)
    
    # Prepare transforms
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    
    descriptors = []
    
    for filepath in tqdm(filenames, desc="Computing global descriptors"):
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            continue
        
        try:
            # Load and preprocess image
            img = cv2.imread(filepath)
            if img is None:
                logger.error(f"Failed to load image: {filepath}")
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512, 512))
            img_pil = Image.fromarray(img)
            img_tensor = transform(img_pil).unsqueeze(0).to(device)
            
            # Extract features
            with torch.no_grad():
                desc = model.forward_features(img_tensor).mean(dim=(-1, 2))
                desc = desc.view(1, -1)
                
                # TTA with horizontal flip
                desc_lr = model.forward_features(img_tensor.flip(-1)).mean(dim=(-1, 2))
                desc_lr = desc_lr.view(1, -1)
                
                # Average and normalize
                desc_avg = F.normalize((desc + desc_lr) / 2, dim=1, p=2)
                descriptors.append(desc_avg.cpu())
                
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            continue
    
    if descriptors:
        return torch.cat(descriptors, dim=0)
    else:
        return torch.tensor([])


def get_image_pairs_shortlist(filenames: List[str],
                            similarity_threshold: float = 0.5,
                            min_pairs_per_image: int = 35,
                            exhaustive_if_less_than: int = 20,
                            device: Optional[torch.device] = None,
                            model_name: str = 'tf_efficientnet_b7',
                            model_path: str = './efficientnet/tf_efficientnet_b7.pth') -> List[Tuple[int, int]]:
    """
    Generate image pairs based on global descriptor similarity.
    
    Uses global image descriptors to find similar images and create
    a shortlist of pairs for matching, avoiding exhaustive matching
    for large datasets.
    """
    num_images = len(filenames)
    
    # Use exhaustive matching for small datasets
    if num_images <= exhaustive_if_less_than:
        logger.info(f"Using exhaustive matching for {num_images} images")
        return get_image_pairs_exhaustive(filenames)
    
    logger.info(f"Computing image pairs using global descriptors for {num_images} images")
    
    # Compute global descriptors
    descriptors = compute_global_descriptors(filenames, model_name, model_path, device)
    
    if len(descriptors) == 0:
        logger.error("No descriptors computed, falling back to exhaustive matching")
        return get_image_pairs_exhaustive(filenames)
    
    # Compute pairwise distances
    distances = torch.cdist(descriptors, descriptors, p=2).cpu().numpy()
    
    # Generate pairs based on similarity
    pairs = set()
    indices = np.arange(num_images)
    
    for i in range(num_images - 1):
        # Find similar images
        similar_mask = distances[i] <= similarity_threshold
        similar_indices = indices[similar_mask]
        
        # Ensure minimum pairs per image
        if len(similar_indices) < min_pairs_per_image:
            similar_indices = np.argsort(distances[i])[:min_pairs_per_image]
        
        # Add pairs
        for j in similar_indices:
            if i != j and distances[i, j] < 1000:  # Sanity check
                pairs.add(tuple(sorted((i, int(j)))))
    
    pairs_list = sorted(list(pairs))
    logger.info(f"Generated {len(pairs_list)} image pairs")
    
    return pairs_list


class ImagePairSelector:
    """
    Select image pairs for matching using global descriptors.
    
    This class uses pretrained CNN models (e.g., EfficientNet, ResNet) to:
    - Extract global descriptors from images
    - Compute similarity between images
    - Select optimal pairs for 3D reconstruction
    
    The pair selection strategy balances between:
    - Selecting similar images (high chance of overlap)
    - Ensuring sufficient baseline for 3D reconstruction
    - Computational efficiency
    """
    
    def __init__(
        self,
        device: torch.device,
        config: Optional[Dict] = None,
        model_name: str = 'tf_efficientnet_b7'
    ):
        """
        Initialize the image pair selector.
        
        Args:
            device: PyTorch device
            config: Configuration dictionary
            model_name: Name of the model to use for descriptors
        """
        self.device = device
        self.config = config or self._get_default_config()
        self.model_name = model_name
        self._setup_model()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'min_pairs_per_image': 35,
            'similarity_threshold': 0.5,
            'exhaustive_if_less_than': 20,
            'model_path': './efficientnet/tf_efficientnet_b7.pth'
        }
    
    def _setup_model(self):
        """Setup the feature extraction model."""
        logger.info(f"Setting up {self.model_name} for global descriptors...")
    
    def select_pairs(
        self,
        image_paths: List[str],
        verbose: bool = True
    ) -> List[Tuple[int, int]]:
        """
        Select image pairs for matching.
        
        Args:
            image_paths: List of image file paths
            verbose: Show progress bar
        
        Returns:
            List of (idx1, idx2) pairs to match
        """
        return get_image_pairs_shortlist(
            image_paths,
            similarity_threshold=self.config['similarity_threshold'],
            min_pairs_per_image=self.config['min_pairs_per_image'],
            exhaustive_if_less_than=self.config['exhaustive_if_less_than'],
            device=self.device,
            model_name=self.model_name,
            model_path=self.config['model_path']
        )