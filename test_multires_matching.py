#!/usr/bin/env python3
"""
Test multi-resolution matching improvements.
"""

import numpy as np
from pathlib import Path
import logging
from vision3d import Vision3DPipeline
from vision3d.models.loftr import LoFTRMatcher
import torch
import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_resolution_comparison():
    """Compare matching at different resolutions."""
    
    # Get test images
    image_dir = Path("examples/images")
    images = list(image_dir.glob("*.png"))[:2]
    
    if len(images) < 2:
        logger.error("Need at least 2 images for testing")
        return
        
    img1_path = str(images[0])
    img2_path = str(images[1])
    
    logger.info(f"Testing with images:")
    logger.info(f"  Image 1: {images[0].name}")
    logger.info(f"  Image 2: {images[1].name}")
    
    # Initialize matcher
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matcher = LoFTRMatcher(device=device)
    
    # Test 1: Single resolution (640 - old default)
    logger.info("\n=== Test 1: Low resolution (640px) ===")
    start = time.time()
    mkpts1_640, mkpts2_640, conf_640 = matcher.match_pair(
        img1_path, img2_path, resize=640
    )
    time_640 = time.time() - start
    logger.info(f"Resolution 640: {len(mkpts1_640)} matches in {time_640:.2f}s")
    logger.info(f"Average confidence: {conf_640.mean():.3f}")
    
    # Test 2: Single resolution (1024 - new default)
    logger.info("\n=== Test 2: Medium resolution (1024px) ===")
    start = time.time()
    mkpts1_1024, mkpts2_1024, conf_1024 = matcher.match_pair(
        img1_path, img2_path, resize=1024
    )
    time_1024 = time.time() - start
    logger.info(f"Resolution 1024: {len(mkpts1_1024)} matches in {time_1024:.2f}s")
    logger.info(f"Average confidence: {conf_1024.mean():.3f}")
    
    # Test 3: Single resolution (1440)
    logger.info("\n=== Test 3: High resolution (1440px) ===")
    start = time.time()
    mkpts1_1440, mkpts2_1440, conf_1440 = matcher.match_pair(
        img1_path, img2_path, resize=1440
    )
    time_1440 = time.time() - start
    logger.info(f"Resolution 1440: {len(mkpts1_1440)} matches in {time_1440:.2f}s")
    logger.info(f"Average confidence: {conf_1440.mean():.3f}")
    
    # Test 4: Multi-resolution matching
    logger.info("\n=== Test 4: Multi-resolution (1024 + 1440) ===")
    start = time.time()
    mkpts1_multi, mkpts2_multi, conf_multi = matcher.match_pair_multires(
        img1_path, img2_path, 
        resolutions=[1024, 1440],
        bidirectional=True
    )
    time_multi = time.time() - start
    logger.info(f"Multi-resolution: {len(mkpts1_multi)} matches in {time_multi:.2f}s")
    logger.info(f"Average confidence: {conf_multi.mean():.3f}")
    
    # Summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"640px:  {len(mkpts1_640):4d} matches, {time_640:5.2f}s, avg conf: {conf_640.mean():.3f}")
    logger.info(f"1024px: {len(mkpts1_1024):4d} matches, {time_1024:5.2f}s, avg conf: {conf_1024.mean():.3f}")
    logger.info(f"1440px: {len(mkpts1_1440):4d} matches, {time_1440:5.2f}s, avg conf: {conf_1440.mean():.3f}")
    logger.info(f"Multi:  {len(mkpts1_multi):4d} matches, {time_multi:5.2f}s, avg conf: {conf_multi.mean():.3f}")
    
    # Calculate improvements
    improvement_640_to_1024 = (len(mkpts1_1024) - len(mkpts1_640)) / max(1, len(mkpts1_640)) * 100
    improvement_640_to_multi = (len(mkpts1_multi) - len(mkpts1_640)) / max(1, len(mkpts1_640)) * 100
    
    logger.info(f"\nImprovement from 640 to 1024: {improvement_640_to_1024:+.1f}%")
    logger.info(f"Improvement from 640 to multi-res: {improvement_640_to_multi:+.1f}%")
    
    # Visualize matches
    try:
        import cv2
        from vision3d.utils.visualization import visualize_matches
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        # Visualize 640 vs multi-resolution
        vis_640 = visualize_matches(
            img1, img2, mkpts1_640, mkpts2_640, conf_640,
            max_matches=200,
            save_path="output/matches_640px.jpg"
        )
        
        vis_multi = visualize_matches(
            img1, img2, mkpts1_multi, mkpts2_multi, conf_multi,
            max_matches=200,
            save_path="output/matches_multires.jpg"
        )
        
        logger.info("\nSaved visualizations to output/")
        
    except Exception as e:
        logger.warning(f"Could not create visualizations: {e}")


def test_full_pipeline():
    """Test the full pipeline with multi-resolution matching."""
    
    logger.info("\n=== Testing full pipeline with multi-resolution ===")
    
    # Get test images
    image_dir = Path("examples/images")
    image_paths = [str(p) for p in image_dir.glob("*.png")][:4]
    
    if len(image_paths) < 3:
        logger.error("Need at least 3 images for reconstruction")
        return
    
    # Create pipeline with multi-resolution config
    pipeline = Vision3DPipeline(
        matcher_type='loftr',
        config={
            'loftr': {
                'image_resize': 1024,
                'multi_resolution': [1024, 1440],
                'confidence_threshold': 0.3,
                'use_tta': True
            }
        }
    )
    
    # Run reconstruction
    logger.info(f"Running reconstruction with {len(image_paths)} images")
    start = time.time()
    
    reconstruction = pipeline.reconstruct(
        image_paths,
        output_dir="output_multires"
    )
    
    elapsed = time.time() - start
    
    logger.info(f"\nReconstruction completed in {elapsed:.2f}s")
    logger.info(f"Registered images: {reconstruction['num_images']}")
    logger.info(f"3D points: {reconstruction['num_points3D']}")
    
    if reconstruction['statistics']:
        stats = reconstruction['statistics']
        logger.info(f"Mean reprojection error: {stats['mean_reprojection_error']:.2f} pixels")
        logger.info(f"Mean track length: {stats['mean_track_length']:.1f}")


if __name__ == "__main__":
    logger.info("Testing Vision3D multi-resolution matching improvements")
    logger.info("=" * 60)
    
    # Run tests
    test_resolution_comparison()
    test_full_pipeline()