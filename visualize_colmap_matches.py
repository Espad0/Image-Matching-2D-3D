#!/usr/bin/env python3
"""
Visualize matches from COLMAP reconstruction database.
"""

import numpy as np
import cv2
import pycolmap
from pathlib import Path
import matplotlib.pyplot as plt
from vision3d.utils.visualization import visualize_matches
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_matches_from_database(database_path, output_dir):
    """Extract and visualize matches from COLMAP database."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open database
    db = pycolmap.Database(database_path)
    
    # Get all images
    images = db.read_all_images()
    logger.info(f"Found {len(images)} images in database")
    
    # Get all image pairs with geometries
    image_pairs = []
    for img1_id in range(1, len(images) + 1):
        for img2_id in range(img1_id + 1, len(images) + 1):
            if db.exists_two_view_geometry(img1_id, img2_id):
                two_view_geom = db.read_two_view_geometry(img1_id, img2_id)
                image_pairs.append((img1_id, img2_id, two_view_geom))
    
    logger.info(f"Found {len(image_pairs)} image pairs with matches")
    
    # Process each image pair
    for pair_idx, (img1_id, img2_id, two_view_geom) in enumerate(image_pairs):
        
        # Get image info
        img1_info = images[img1_id]
        img2_info = images[img2_id]
        
        logger.info(f"\nProcessing pair {pair_idx + 1}/{len(image_pairs)}")
        logger.info(f"  Image 1: {img1_info.name} (ID: {img1_id})")
        logger.info(f"  Image 2: {img2_info.name} (ID: {img2_id})")
        
        # Load images
        image_dir = Path("examples/images")  # Adjust if needed
        img1_path = image_dir / img1_info.name
        img2_path = image_dir / img2_info.name
        
        if not img1_path.exists() or not img2_path.exists():
            logger.warning(f"  Skipping - images not found")
            continue
            
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        
        # Get keypoints
        kpts1 = db.read_keypoints(img1_id)
        kpts2 = db.read_keypoints(img2_id)
        
        # Get matches from two-view geometry
        matches = two_view_geom.matches
        inlier_matches = two_view_geom.inlier_matches
        
        logger.info(f"  Total matches: {matches.shape[0]}")
        logger.info(f"  Inlier matches: {np.sum(inlier_matches)}")
        
        if matches.shape[0] == 0:
            continue
            
        # Extract matched keypoints
        valid_matches = matches[matches[:, 0] >= 0]
        mkpts1 = kpts1[valid_matches[:, 0]]
        mkpts2 = kpts2[valid_matches[:, 1]]
        
        # Get inliers
        inlier_mask = inlier_matches[matches[:, 0] >= 0]
        
        # Visualize all matches
        vis_all = visualize_matches(
            img1, img2, mkpts1, mkpts2,
            confidence=inlier_mask.astype(float),
            max_matches=500,
            save_path=str(output_dir / f"matches_pair{pair_idx}_all.jpg")
        )
        
        # Visualize only inliers
        if np.sum(inlier_mask) > 0:
            vis_inliers = visualize_matches(
                img1, img2, 
                mkpts1[inlier_mask], 
                mkpts2[inlier_mask],
                max_matches=500,
                save_path=str(output_dir / f"matches_pair{pair_idx}_inliers.jpg")
            )
            
        # Save match statistics
        stats_path = output_dir / f"match_stats_pair{pair_idx}.txt"
        with open(stats_path, 'w') as f:
            f.write(f"Image pair: {img1_info.name} <-> {img2_info.name}\n")
            f.write(f"Total matches: {matches.shape[0]}\n")
            f.write(f"Inlier matches: {np.sum(inlier_matches)}\n")
            f.write(f"Inlier ratio: {np.sum(inlier_matches) / max(1, matches.shape[0]):.2%}\n")
            
            # If we have the fundamental matrix
            if hasattr(two_view_geom, 'F'):
                f.write(f"\nFundamental matrix:\n{two_view_geom.F}\n")
                
    logger.info(f"\nVisualization complete! Check {output_dir} for results.")
    

def main():
    database_path = "output/colmap.db"
    output_dir = "output/match_visualizations"
    
    if not Path(database_path).exists():
        logger.error(f"Database not found: {database_path}")
        return
        
    extract_matches_from_database(database_path, output_dir)


if __name__ == "__main__":
    main()