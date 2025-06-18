"""
Advanced matching example with custom configuration.

This example demonstrates:
- Custom matcher configuration
- Multi-scale matching
- Test-time augmentation
- Match visualization
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from vision3d.models import LoFTRMatcher, SuperGlueMatcher
from vision3d.utils.visualization import visualize_matches

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def match_with_loftr_multiscale(img1_path, img2_path, device='cuda'):
    """
    Demonstrate multi-scale matching with LoFTR.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        device: Computing device
    
    Returns:
        Matched keypoints and confidences
    """
    logger.info("Initializing LoFTR matcher...")
    
    # Custom LoFTR configuration
    config = {
        'image_resize': 1024,
        'confidence_threshold': 0.2,
        'use_tta': True,
        'tta_variants': ['orig', 'flip_lr', 'rot_90'],
    }
    
    matcher = LoFTRMatcher(device=device, config=config)
    
    # Match at multiple scales
    logger.info("Matching at multiple scales...")
    scales = [640, 1024, 1440]
    all_matches = []
    
    for scale in scales:
        kpts1, kpts2, conf = matcher.match_pair(
            img1_path, img2_path, resize=scale
        )
        all_matches.append({
            'scale': scale,
            'kpts1': kpts1,
            'kpts2': kpts2,
            'conf': conf,
            'num_matches': len(kpts1)
        })
        logger.info(f"  Scale {scale}: {len(kpts1)} matches")
    
    # Combine matches from all scales
    kpts1 = np.vstack([m['kpts1'] for m in all_matches])
    kpts2 = np.vstack([m['kpts2'] for m in all_matches])
    conf = np.hstack([m['conf'] for m in all_matches])
    
    # Remove duplicates
    kpts1, kpts2, conf = matcher._remove_duplicates(kpts1, kpts2, conf)
    
    logger.info(f"Total unique matches: {len(kpts1)}")
    
    return kpts1, kpts2, conf, all_matches


def match_with_superglue_highres(img1_path, img2_path, device='cuda'):
    """
    Demonstrate high-resolution matching with SuperGlue.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        device: Computing device
    
    Returns:
        Matched keypoints and confidences
    """
    logger.info("Initializing SuperGlue matcher...")
    
    # Custom SuperGlue configuration for high-resolution matching
    config = {
        'superpoint': {
            'nms_radius': 3,
            'keypoint_threshold': 0.003,  # Lower threshold for more keypoints
            'max_keypoints': 8192,        # More keypoints
        },
        'superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 100,
            'match_threshold': 0.15,      # Stricter matching
        },
        'confidence_threshold': 0.2,
        'use_tta': True,
        'tta_groups': [
            ('orig', 'orig'),
            ('flip_lr', 'flip_lr'),
            ('eqhist', 'eqhist'),  # Histogram equalization
        ],
    }
    
    matcher = SuperGlueMatcher(device=device, config=config)
    
    logger.info("Performing high-resolution matching...")
    kpts1, kpts2, conf = matcher.match_with_high_resolution(
        img1_path, img2_path, max_keypoints=8192
    )
    
    logger.info(f"Found {len(kpts1)} high-quality matches")
    
    return kpts1, kpts2, conf


def compare_matchers(img1_path, img2_path, output_dir='./output'):
    """
    Compare different matching strategies.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load images
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    # Run different matchers
    results = {}
    
    # LoFTR multi-scale
    logger.info("\n1. LoFTR Multi-scale Matching")
    kpts1_loftr, kpts2_loftr, conf_loftr, _ = match_with_loftr_multiscale(
        img1_path, img2_path
    )
    results['loftr_multiscale'] = {
        'kpts1': kpts1_loftr,
        'kpts2': kpts2_loftr,
        'conf': conf_loftr
    }
    
    # SuperGlue high-resolution
    logger.info("\n2. SuperGlue High-resolution Matching")
    kpts1_sg, kpts2_sg, conf_sg = match_with_superglue_highres(
        img1_path, img2_path
    )
    results['superglue_highres'] = {
        'kpts1': kpts1_sg,
        'kpts2': kpts2_sg,
        'conf': conf_sg
    }
    
    # Visualize results
    logger.info("\nVisualizing results...")
    
    for method, data in results.items():
        # Create visualization
        fig = plt.figure(figsize=(20, 10))
        
        # Show matches
        matched_img = visualize_matches(
            img1, img2,
            data['kpts1'], data['kpts2'],
            data['conf']
        )
        
        plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
        plt.title(f'{method} - {len(data["kpts1"])} matches')
        plt.axis('off')
        
        # Save visualization
        output_path = output_dir / f'{method}_matches.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved {output_path}")
    
    # Print comparison statistics
    logger.info("\nComparison Statistics:")
    for method, data in results.items():
        conf = data['conf']
        logger.info(f"\n{method}:")
        logger.info(f"  - Total matches: {len(conf)}")
        logger.info(f"  - Mean confidence: {np.mean(conf):.3f}")
        logger.info(f"  - Median confidence: {np.median(conf):.3f}")
        logger.info(f"  - High-confidence matches (>0.5): {np.sum(conf > 0.5)}")


def main():
    """Run advanced matching examples."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced matching examples')
    parser.add_argument('--img1', type=str, required=True, help='First image')
    parser.add_argument('--img2', type=str, required=True, help='Second image')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Computing device')
    args = parser.parse_args()
    
    # Run comparison
    compare_matchers(args.img1, args.img2, args.output)


if __name__ == '__main__':
    main()