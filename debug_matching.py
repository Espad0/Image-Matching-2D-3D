"""
Debug script for visualizing and analyzing LoFTR matching results.
This script helps diagnose why matches might be failing between image pairs.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import torch
from vision3d.models import LoFTRMatcher
from vision3d.utils.visualization import visualize_matches
import json

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def visualize_keypoints(img, keypoints, title, save_path):
    """Visualize keypoints on image."""
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if len(keypoints) > 0:
        plt.scatter(keypoints[:, 0], keypoints[:, 1], c='red', s=10, alpha=0.5)
        plt.title(f"{title} - {len(keypoints)} keypoints")
    else:
        plt.title(f"{title} - No keypoints found")
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved keypoint visualization to {save_path}")

def analyze_matches(mkpts1, mkpts2, mconf, img1, img2, output_dir):
    """Analyze match statistics and distributions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'num_matches': len(mkpts1),
        'confidence': {
            'mean': float(mconf.mean()) if len(mconf) > 0 else 0,
            'std': float(mconf.std()) if len(mconf) > 0 else 0,
            'min': float(mconf.min()) if len(mconf) > 0 else 0,
            'max': float(mconf.max()) if len(mconf) > 0 else 0
        },
        'spatial_distribution': {}
    }
    
    if len(mkpts1) > 0:
        # Analyze spatial distribution
        img1_h, img1_w = img1.shape[:2]
        img2_h, img2_w = img2.shape[:2]
        
        # Divide image into quadrants
        quadrants1 = np.zeros(4)
        quadrants2 = np.zeros(4)
        
        for pt in mkpts1:
            x, y = pt
            if x < img1_w/2 and y < img1_h/2:
                quadrants1[0] += 1
            elif x >= img1_w/2 and y < img1_h/2:
                quadrants1[1] += 1
            elif x < img1_w/2 and y >= img1_h/2:
                quadrants1[2] += 1
            else:
                quadrants1[3] += 1
        
        for pt in mkpts2:
            x, y = pt
            if x < img2_w/2 and y < img2_h/2:
                quadrants2[0] += 1
            elif x >= img2_w/2 and y < img2_h/2:
                quadrants2[1] += 1
            elif x < img2_w/2 and y >= img2_h/2:
                quadrants2[2] += 1
            else:
                quadrants2[3] += 1
        
        stats['spatial_distribution'] = {
            'image1_quadrants': quadrants1.tolist(),
            'image2_quadrants': quadrants2.tolist()
        }
        
        # Plot confidence distribution
        plt.figure(figsize=(10, 6))
        plt.hist(mconf, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Confidence Score')
        plt.ylabel('Number of Matches')
        plt.title(f'Match Confidence Distribution (Total: {len(mconf)})')
        plt.axvline(mconf.mean(), color='red', linestyle='--', label=f'Mean: {mconf.mean():.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'confidence_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot match displacement vectors
        plt.figure(figsize=(10, 10))
        displacements = mkpts2 - mkpts1
        plt.quiver(mkpts1[:, 0], mkpts1[:, 1], 
                   displacements[:, 0], displacements[:, 1],
                   mconf, cmap='viridis', alpha=0.6)
        plt.colorbar(label='Confidence')
        plt.xlim(0, img1_w)
        plt.ylim(img1_h, 0)
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title('Match Displacement Vectors')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'displacement_vectors.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Save statistics
    with open(output_dir / 'match_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Match statistics saved to {output_dir / 'match_statistics.json'}")
    return stats

def debug_matching(img1_path, img2_path, output_dir='debug_output'):
    """Run comprehensive matching debug analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting debug matching for:")
    logger.info(f"  Image 1: {img1_path}")
    logger.info(f"  Image 2: {img2_path}")
    
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        logger.error("Failed to load one or both images!")
        return
    
    logger.info(f"Image 1 shape: {img1.shape}")
    logger.info(f"Image 2 shape: {img2.shape}")
    
    # Save input images
    cv2.imwrite(str(output_dir / 'input_img1.jpg'), img1)
    cv2.imwrite(str(output_dir / 'input_img2.jpg'), img2)
    
    # Initialize matcher with different configurations
    configs = [
        {
            'name': 'default',
            'config': {
                'image_resize': 1024,
                'confidence_threshold': 0.2,
                'use_tta': True,
                'tta_variants': ['orig', 'flip_lr']
            }
        },
        {
            'name': 'low_threshold',
            'config': {
                'image_resize': 1024,
                'confidence_threshold': 0.1,
                'use_tta': True,
                'tta_variants': ['orig', 'flip_lr']
            }
        },
        {
            'name': 'no_tta',
            'config': {
                'image_resize': 1024,
                'confidence_threshold': 0.2,
                'use_tta': False
            }
        },
        {
            'name': 'high_res',
            'config': {
                'image_resize': 1440,
                'confidence_threshold': 0.2,
                'use_tta': True,
                'tta_variants': ['orig', 'flip_lr']
            }
        }
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    all_results = {}
    
    for cfg in configs:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing configuration: {cfg['name']}")
        logger.info(f"Config: {cfg['config']}")
        
        try:
            # Create output directory for this config
            config_dir = output_dir / cfg['name']
            config_dir.mkdir(exist_ok=True)
            
            # Initialize matcher
            matcher = LoFTRMatcher(device=device, config=cfg['config'])
            
            # Perform matching
            mkpts1, mkpts2, mconf = matcher.match_pair(img1_path, img2_path)
            
            logger.info(f"Found {len(mkpts1)} matches")
            if len(mkpts1) > 0:
                logger.info(f"Confidence range: [{mconf.min():.3f}, {mconf.max():.3f}]")
                logger.info(f"Mean confidence: {mconf.mean():.3f}")
            
            # Visualize keypoints
            visualize_keypoints(img1, mkpts1, f"Image 1 - {cfg['name']}", 
                               config_dir / 'keypoints_img1.png')
            visualize_keypoints(img2, mkpts2, f"Image 2 - {cfg['name']}", 
                               config_dir / 'keypoints_img2.png')
            
            # Draw matches
            if len(mkpts1) > 0:
                # Draw all matches
                match_img_all = visualize_matches(img1, img2, mkpts1, mkpts2, mconf, 
                                                  save_path=str(config_dir / 'matches_all.png'))
                
                # Draw top confidence matches
                if len(mkpts1) > 20:
                    top_indices = np.argsort(mconf)[-20:]  # Top 20 matches
                    match_img_top = visualize_matches(
                        img1, img2, 
                        mkpts1[top_indices], 
                        mkpts2[top_indices], 
                        mconf[top_indices],
                        save_path=str(config_dir / 'matches_top20.png')
                    )
            
            # Analyze matches
            stats = analyze_matches(mkpts1, mkpts2, mconf, img1, img2, config_dir)
            all_results[cfg['name']] = stats
            
        except Exception as e:
            logger.error(f"Error with configuration {cfg['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
            all_results[cfg['name']] = {'error': str(e)}
    
    # Compare results across configurations
    logger.info(f"\n{'='*50}")
    logger.info("Summary of all configurations:")
    for name, stats in all_results.items():
        if 'error' in stats:
            logger.info(f"{name}: ERROR - {stats['error']}")
        else:
            logger.info(f"{name}: {stats['num_matches']} matches, "
                       f"mean conf: {stats['confidence']['mean']:.3f}")
    
    # Save overall summary
    with open(output_dir / 'debug_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nDebug output saved to: {output_dir}")
    return all_results

if __name__ == "__main__":
    # Test with two example images
    img1 = "examples/3DOM_FBK_IMG_1516.png"
    img2 = "examples/3DOM_FBK_IMG_1520.png"
    
    results = debug_matching(img1, img2, "debug_output")