"""
Vision3D: State-of-the-Art 3D Reconstruction from Images

A production-ready implementation of advanced computer vision techniques for 3D reconstruction,
featuring LoFTR, SuperGlue, and COLMAP integration.

This package provides a complete pipeline for:
- Dense and sparse feature matching
- Multi-view geometry estimation
- 3D reconstruction from image collections
- Production-ready code with best practices
"""

__version__ = "1.0.0"
__author__ = "Andrej Nesterov"

from .core.pipeline import Vision3DPipeline
from .models import LoFTRMatcher, SuperGlueMatcher
from .utils.visualization import visualize_matches, visualize_reconstruction

__all__ = [
    "Vision3DPipeline",
    "LoFTRMatcher",
    "SuperGlueMatcher",
    "visualize_matches",
    "visualize_reconstruction"
]