#!/usr/bin/env python3
"""
Test all IMC-2023 inspired upgrades to Vision3D.
"""

import numpy as np
from pathlib import Path
import logging
import time
from vision3d import Vision3DPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_all_upgrades():
    """Test all the implemented upgrades."""
    
    # Get test images
    image_dir = Path("examples/images")
    image_paths = [str(p) for p in image_dir.glob("*.png")][:5]
    
    if len(image_paths) < 3:
        logger.error("Need at least 3 images for testing")
        return
    
    logger.info(f"Testing with {len(image_paths)} images")
    
    # Test configurations
    configs = [
        {
            'name': 'Baseline (old defaults)',
            'matcher_type': 'loftr',
            'config': {
                'matching': {
                    'image_resize': 640,
                    'use_tta': False,
                    'confidence_threshold': 0.3
                },
                'reconstruction': {
                    'min_model_size': 10,
                    'ba_local_max_refinements': 3,
                    'ba_global_max_refinements': 10
                }
            }
        },
        {
            'name': 'With TTA only',
            'matcher_type': 'loftr',
            'config': {
                'matching': {
                    'image_resize': 640,
                    'use_tta': True,
                    'tta_variants': ['orig', 'flip_lr'],
                    'confidence_threshold': 0.3
                }
            }
        },
        {
            'name': 'Higher resolution (1024px)',
            'matcher_type': 'loftr',
            'config': {
                'matching': {
                    'image_resize': 1024,
                    'use_tta': True,
                    'confidence_threshold': 0.3
                }
            }
        },
        {
            'name': 'Multi-method (IMC-2023 approach)',
            'matcher_type': 'multi',
            'config': {
                'matching': {
                    'image_resize': 1024,
                    'confidence_threshold': 0.3
                }
            }
        }
    ]
    
    results = []
    
    for cfg in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {cfg['name']}")
        logger.info(f"{'='*60}")
        
        # Create pipeline
        pipeline = Vision3DPipeline(
            matcher_type=cfg['matcher_type'],
            config=cfg.get('config', {})
        )
        
        # Run reconstruction
        start_time = time.time()
        try:
            reconstruction = pipeline.reconstruct(
                image_paths,
                output_dir=f"output_test_{cfg['name'].replace(' ', '_').lower()}",
                verbose=False
            )
            elapsed = time.time() - start_time
            
            # Collect results
            result = {
                'name': cfg['name'],
                'success': True,
                'time': elapsed,
                'num_images': reconstruction['num_images'],
                'num_points': reconstruction['num_points3D'],
                'mean_reproj_error': reconstruction['statistics'].get('mean_reprojection_error', -1),
                'mean_track_length': reconstruction['statistics'].get('mean_track_length', -1)
            }
            
        except Exception as e:
            logger.error(f"Failed: {e}")
            result = {
                'name': cfg['name'],
                'success': False,
                'time': time.time() - start_time,
                'num_images': 0,
                'num_points': 0,
                'mean_reproj_error': -1,
                'mean_track_length': -1
            }
        
        results.append(result)
        logger.info(f"Result: {result}")
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY OF IMC-2023 UPGRADES")
    logger.info(f"{'='*80}")
    logger.info(f"{'Configuration':<30} {'Success':<10} {'Images':<10} {'Points':<10} {'Time':<10}")
    logger.info(f"{'-'*80}")
    
    for r in results:
        logger.info(
            f"{r['name']:<30} "
            f"{'Yes' if r['success'] else 'No':<10} "
            f"{r['num_images']:<10} "
            f"{r['num_points']:<10} "
            f"{r['time']:.2f}s"
        )
    
    # Calculate improvements
    if results[0]['success'] and results[-1]['success']:
        baseline_points = results[0]['num_points']
        imc_points = results[-1]['num_points']
        if baseline_points > 0:
            improvement = (imc_points - baseline_points) / baseline_points * 100
            logger.info(f"\nIMC-2023 approach improvement: {improvement:+.1f}% more 3D points")


def test_matching_improvements():
    """Test just the matching improvements."""
    
    logger.info("\n" + "="*60)
    logger.info("TESTING MATCHING IMPROVEMENTS")
    logger.info("="*60)
    
    from vision3d.models.loftr import LoFTRMatcher
    from vision3d.models.multi_method_matcher import MultiMethodMatcher
    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get test images
    image_dir = Path("examples/images")
    images = list(image_dir.glob("*.png"))[:2]
    
    if len(images) < 2:
        logger.error("Need at least 2 images")
        return
    
    img1, img2 = str(images[0]), str(images[1])
    
    # Test 1: Baseline LoFTR
    logger.info("\n1. Baseline LoFTR (640px, no TTA)")
    matcher1 = LoFTRMatcher(device=device, config={
        'image_resize': 640,
        'use_tta': False,
        'confidence_threshold': 0.3
    })
    mkpts1, mkpts2, conf = matcher1.match_pair(img1, img2)
    baseline_matches = len(mkpts1)
    logger.info(f"   Matches: {baseline_matches}")
    
    # Test 2: LoFTR with TTA
    logger.info("\n2. LoFTR with TTA (640px)")
    matcher2 = LoFTRMatcher(device=device, config={
        'image_resize': 640,
        'use_tta': True,
        'confidence_threshold': 0.3
    })
    mkpts1, mkpts2, conf = matcher2.match_pair(img1, img2)
    tta_matches = len(mkpts1)
    logger.info(f"   Matches: {tta_matches} (+{(tta_matches-baseline_matches)/baseline_matches*100:.1f}%)")
    
    # Test 3: Multi-resolution
    logger.info("\n3. Multi-resolution LoFTR (1024+1440px, TTA, bidirectional)")
    mkpts1, mkpts2, conf = matcher2.match_pair_multires(
        img1, img2,
        resolutions=[1024, 1440],
        bidirectional=True
    )
    multires_matches = len(mkpts1)
    logger.info(f"   Matches: {multires_matches} (+{(multires_matches-baseline_matches)/baseline_matches*100:.1f}%)")
    
    # Test 4: Multi-method
    logger.info("\n4. Multi-method (LoFTR + SuperGlue)")
    matcher3 = MultiMethodMatcher(device=device)
    mkpts1, mkpts2, conf = matcher3.match_pair(img1, img2, num_pairs=10)
    multi_matches = len(mkpts1)
    logger.info(f"   Matches: {multi_matches} (+{(multi_matches-baseline_matches)/baseline_matches*100:.1f}%)")


if __name__ == "__main__":
    # Run tests
    test_matching_improvements()
    test_all_upgrades()