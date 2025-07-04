#!/usr/bin/env python3
"""
Quick test to show the impact of higher resolution matching.
"""

import numpy as np
from pathlib import Path
import logging
from vision3d.models.loftr import LoFTRMatcher
import torch
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_single_pair():
    """Test a single image pair at different resolutions."""
    
    # Get test images
    image_dir = Path("examples/images")
    images = list(image_dir.glob("*.png"))[:2]
    
    if len(images) < 2:
        logger.error("Need at least 2 images for testing")
        return
        
    img1_path = str(images[0])
    img2_path = str(images[1])
    
    logger.info(f"Testing with: {images[0].name} <-> {images[1].name}")
    
    # Initialize matcher with TTA disabled for faster testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'image_resize': 1024,
        'confidence_threshold': 0.3,
        'use_tta': False  # Disable TTA for speed
    }
    matcher = LoFTRMatcher(device=device, config=config)
    
    # Test different resolutions
    resolutions = [640, 1024, 1440]
    results = {}
    
    for res in resolutions:
        logger.info(f"\nTesting resolution: {res}px")
        start = time.time()
        
        mkpts1, mkpts2, conf = matcher.match_pair(
            img1_path, img2_path, resize=res
        )
        
        elapsed = time.time() - start
        results[res] = {
            'matches': len(mkpts1),
            'time': elapsed,
            'confidence': conf.mean() if len(conf) > 0 else 0
        }
        
        logger.info(f"  Matches: {len(mkpts1)}")
        logger.info(f"  Time: {elapsed:.2f}s")
        logger.info(f"  Avg confidence: {results[res]['confidence']:.3f}")
    
    # Show improvements
    logger.info("\n=== RESULTS SUMMARY ===")
    logger.info(f"{'Resolution':<12} {'Matches':<10} {'Time':<10} {'Confidence':<12}")
    logger.info("-" * 45)
    
    for res in resolutions:
        r = results[res]
        logger.info(f"{res}px{'':<7} {r['matches']:<10} {r['time']:<10.2f} {r['confidence']:<12.3f}")
    
    # Calculate improvements
    base_matches = results[640]['matches']
    if base_matches > 0:
        logger.info("\n=== IMPROVEMENTS vs 640px ===")
        for res in [1024, 1440]:
            improvement = (results[res]['matches'] - base_matches) / base_matches * 100
            logger.info(f"{res}px: {improvement:+.1f}% more matches")
    
    # Test multi-resolution
    logger.info("\n=== MULTI-RESOLUTION TEST ===")
    start = time.time()
    mkpts1_multi, mkpts2_multi, conf_multi = matcher.match_pair_multires(
        img1_path, img2_path,
        resolutions=[1024, 1440],
        bidirectional=False  # Disable for speed
    )
    elapsed = time.time() - start
    
    logger.info(f"Multi-resolution matches: {len(mkpts1_multi)}")
    logger.info(f"Time: {elapsed:.2f}s")
    logger.info(f"Avg confidence: {conf_multi.mean():.3f}")
    
    if base_matches > 0:
        improvement = (len(mkpts1_multi) - base_matches) / base_matches * 100
        logger.info(f"Improvement vs 640px: {improvement:+.1f}%")


if __name__ == "__main__":
    test_single_pair()