#!/usr/bin/env python3
"""
Quick test to verify TTA is working correctly.
"""

import numpy as np
from pathlib import Path
import logging
from vision3d.models.loftr import LoFTRMatcher
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tta():
    """Test that TTA is working correctly."""
    
    # Get test images
    image_dir = Path("examples/images")
    images = list(image_dir.glob("*.png"))[:2]
    
    if len(images) < 2:
        logger.error("Need at least 2 images for testing")
        return
    
    img1_path = str(images[0])
    img2_path = str(images[1])
    
    logger.info(f"Testing TTA with {images[0].name} and {images[1].name}")
    
    # Initialize matcher
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Test 1: Without TTA
    logger.info("\n=== Test 1: Without TTA ===")
    matcher_no_tta = LoFTRMatcher(device=device, config={
        'use_tta': False, 
        'image_resize': 640,
        'confidence_threshold': 0.3
    })
    mkpts1_no_tta, mkpts2_no_tta, conf_no_tta = matcher_no_tta.match_pair(img1_path, img2_path)
    logger.info(f"Without TTA: {len(mkpts1_no_tta)} matches")
    
    # Test 2: With TTA
    logger.info("\n=== Test 2: With TTA ===")
    matcher_tta = LoFTRMatcher(device=device, config={
        'use_tta': True, 
        'image_resize': 640,
        'confidence_threshold': 0.3,
        'tta_variants': ['orig', 'flip_lr']
    })
    mkpts1_tta, mkpts2_tta, conf_tta = matcher_tta.match_pair(img1_path, img2_path)
    logger.info(f"With TTA: {len(mkpts1_tta)} matches")
    
    # Show improvement
    improvement = (len(mkpts1_tta) - len(mkpts1_no_tta)) / max(1, len(mkpts1_no_tta)) * 100
    logger.info(f"\nImprovement with TTA: {improvement:+.1f}%")
    
    # Test 3: Multi-resolution with bidirectional
    logger.info("\n=== Test 3: Multi-resolution + Bidirectional ===")
    mkpts1_multi, mkpts2_multi, conf_multi = matcher_tta.match_pair_multires(
        img1_path, img2_path,
        resolutions=[640, 1024],
        bidirectional=True
    )
    logger.info(f"Multi-res + Bidirectional: {len(mkpts1_multi)} matches")
    
    improvement_multi = (len(mkpts1_multi) - len(mkpts1_no_tta)) / max(1, len(mkpts1_no_tta)) * 100
    logger.info(f"Improvement over baseline: {improvement_multi:+.1f}%")

if __name__ == "__main__":
    test_tta()