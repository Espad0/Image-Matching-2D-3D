"""
Vision3D: State-of-the-Art 3D Reconstruction from Images

A production-ready library for 3D scene reconstruction from photographs,
integrating deep learning with classical computer vision.
"""

__version__ = "0.1.0"
__author__ = "Andrej Nesterov"
__license__ = "MIT"

# Import main pipeline class
from .core.pipeline import Vision3DPipeline

# Import key components for advanced usage
from .core.feature_extraction import FeatureExtractor
from .core.reconstruction import ReconstructionEngine

# Import matchers
from .models.base import BaseMatcher
from .models.superglue_matcher import SuperGlueMatcher
from .models.loftr import LoFTRMatcher

# Import utilities
from .utils.image_pairs import ImagePairSelector, get_image_pairs_shortlist
from .utils.colmap_interface import ColmapInterface

__all__ = [
    "Vision3DPipeline",
    "FeatureExtractor",
    "ReconstructionEngine",
    "BaseMatcher",
    "SuperGlueMatcher",
    "LoFTRMatcher",
    "ImagePairSelector",
    "get_image_pairs_shortlist",
    "ColmapInterface",
]