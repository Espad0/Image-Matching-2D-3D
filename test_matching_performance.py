#!/usr/bin/env python3
"""Test script to verify LoFTR matching performance and ensure no hanging."""

import time
import logging
from pathlib import Path
from vision3d.models.loftr import LoFTRMatcher
import torch

# Setup logging with DEBUG level to see detailed progress
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test images
image_paths = [
    "examples/images/3DOM_FBK_IMG_1556.png",
    "examples/images/3DOM_FBK_IMG_1552.png"
]

# Check if images exist
for path in image_paths:
    if not Path(path).exists():
        logger.error(f"Image not found: {path}")
        exit(1)

logger.info("Starting LoFTR matching performance test...")

# Test configurations
configs = [
    {
        "name": "No TTA (fast)",
        "config": {
            'image_resize': 1024,
            'confidence_threshold': 0.2,
            'use_tta': False,
            'max_keypoints': 8192
        }
    },
    {
        "name": "With TTA (slower but more matches)",
        "config": {
            'image_resize': 1024,
            'confidence_threshold': 0.2,
            'use_tta': True,
            'tta_variants': ['orig', 'flip_lr'],
            'max_keypoints': 8192
        }
    },
    {
        "name": "Large image size",
        "config": {
            'image_resize': 1440,
            'confidence_threshold': 0.2,
            'use_tta': False,
            'max_keypoints': 8192
        }
    }
]

# Test with different configurations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

for test_config in configs:
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {test_config['name']}")
    logger.info(f"{'='*60}")
    
    try:
        # Initialize matcher
        start_time = time.time()
        matcher = LoFTRMatcher(device=device, config=test_config['config'])
        init_time = time.time() - start_time
        logger.info(f"Matcher initialization took: {init_time:.2f}s")
        
        # Run matching
        start_time = time.time()
        kpts1, kpts2, conf = matcher.match_pair(image_paths[0], image_paths[1])
        match_time = time.time() - start_time
        
        # Report results
        logger.info(f"Matching completed in: {match_time:.2f}s")
        logger.info(f"Found {len(kpts1)} matches")
        if len(conf) > 0:
            logger.info(f"Confidence range: [{conf.min():.3f}, {conf.max():.3f}]")
            logger.info(f"Mean confidence: {conf.mean():.3f}")
            
            # Count high-confidence matches
            high_conf = (conf > 0.5).sum()
            logger.info(f"High-confidence matches (>0.5): {high_conf}")
        
    except KeyboardInterrupt:
        logger.error("Test interrupted by user (possible hanging detected)")
        break
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

logger.info("\nPerformance test completed!")