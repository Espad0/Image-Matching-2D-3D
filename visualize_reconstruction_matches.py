#!/usr/bin/env python3
"""
Visualize matches from COLMAP reconstruction.
"""

import numpy as np
import cv2
import pycolmap
from pathlib import Path
from vision3d.utils.visualization import visualize_matches
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_reconstruction_matches(sparse_dir, image_dir, output_dir):
    """Visualize matches from a successful COLMAP reconstruction."""
    sparse_dir = Path(sparse_dir)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load reconstruction
    reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    logger.info(f"Loaded reconstruction with {len(reconstruction.images)} images")
    logger.info(f"Total 3D points: {len(reconstruction.points3D)}")
    
    # Get all image pairs that share 3D points
    image_pairs = {}
    
    # For each 3D point, find which images observe it
    for point3d_id, point3d in reconstruction.points3D.items():
        observing_images = list(point3d.track.elements)
        
        # Create pairs from all images that see this point
        for i in range(len(observing_images)):
            for j in range(i + 1, len(observing_images)):
                img1_id = observing_images[i].image_id
                img2_id = observing_images[j].image_id
                pair_key = (min(img1_id, img2_id), max(img1_id, img2_id))
                
                if pair_key not in image_pairs:
                    image_pairs[pair_key] = []
                    
                # Store the 3D point ID and the 2D observations
                image_pairs[pair_key].append({
                    'point3d_id': point3d_id,
                    'img1_point2d_idx': observing_images[i].point2D_idx,
                    'img2_point2d_idx': observing_images[j].point2D_idx
                })
    
    logger.info(f"Found {len(image_pairs)} image pairs with shared points")
    
    # Visualize top pairs with most matches
    sorted_pairs = sorted(image_pairs.items(), key=lambda x: len(x[1]), reverse=True)
    
    for pair_idx, ((img1_id, img2_id), matches) in enumerate(sorted_pairs[:10]):  # Top 10 pairs
        if img1_id not in reconstruction.images or img2_id not in reconstruction.images:
            continue
            
        img1_info = reconstruction.images[img1_id]
        img2_info = reconstruction.images[img2_id]
        
        logger.info(f"\nVisualizing pair {pair_idx + 1}:")
        logger.info(f"  {img1_info.name} <-> {img2_info.name}")
        logger.info(f"  Shared 3D points: {len(matches)}")
        
        # Load images
        img1_path = image_dir / img1_info.name
        img2_path = image_dir / img2_info.name
        
        if not img1_path.exists() or not img2_path.exists():
            logger.warning(f"  Images not found, skipping")
            continue
            
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        
        # Extract matched 2D points
        pts1 = []
        pts2 = []
        
        for match in matches:
            # Get 2D coordinates from the image observations
            point2d_1 = img1_info.points2D[match['img1_point2d_idx']]
            point2d_2 = img2_info.points2D[match['img2_point2d_idx']]
            
            pts1.append(point2d_1.xy)
            pts2.append(point2d_2.xy)
            
        pts1 = np.array(pts1)
        pts2 = np.array(pts2)
        
        # Visualize matches
        vis = visualize_matches(
            img1, img2, pts1, pts2,
            max_matches=500,
            save_path=str(output_dir / f"reconstruction_matches_pair{pair_idx}.jpg")
        )
        
        # Save match info
        info_path = output_dir / f"match_info_pair{pair_idx}.txt"
        with open(info_path, 'w') as f:
            f.write(f"Image pair: {img1_info.name} <-> {img2_info.name}\n")
            f.write(f"Image IDs: {img1_id} <-> {img2_id}\n")
            f.write(f"Shared 3D points: {len(matches)}\n")
            f.write(f"\nCamera 1:\n")
            f.write(f"  Model: {reconstruction.cameras[img1_info.camera_id].model}\n")
            f.write(f"  Params: {reconstruction.cameras[img1_info.camera_id].params}\n")
            f.write(f"\nCamera 2:\n")
            f.write(f"  Model: {reconstruction.cameras[img2_info.camera_id].model}\n")
            f.write(f"  Params: {reconstruction.cameras[img2_info.camera_id].params}\n")
            
    logger.info(f"\nVisualization complete! Check {output_dir}")


def main():
    # Try both reconstruction folders
    for recon_idx in [0, 1]:
        sparse_dir = f"output/sparse/{recon_idx}"
        if Path(sparse_dir).exists() and any(Path(sparse_dir).iterdir()):
            logger.info(f"Using reconstruction from {sparse_dir}")
            visualize_reconstruction_matches(
                sparse_dir,
                "examples/images",
                f"output/reconstruction_{recon_idx}_matches"
            )
            

if __name__ == "__main__":
    main()