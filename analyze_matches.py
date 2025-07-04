#!/usr/bin/env python3
"""Analyze matches before and after reconstruction."""

import pycolmap
import numpy as np
from pathlib import Path

def analyze_matches():
    # Load database
    db = pycolmap.Database('output/colmap.db')
    images = db.read_all_images()
    
    print("=== RAW MATCHES IN DATABASE ===")
    print(f"Total images: {len(images)}")
    
    # Convert to dict if needed
    if isinstance(images, list):
        images = {i+1: img for i, img in enumerate(images)}
    
    # Check all pairs
    raw_matches = {}
    for img1_id, img1 in images.items():
        for img2_id, img2 in images.items():
            if img1_id >= img2_id:
                continue
                
            matches = db.read_matches(img1_id, img2_id)
            if len(matches) > 0:
                pair_key = (img1.name, img2.name)
                raw_matches[pair_key] = len(matches)
                print(f"{img1.name} <-> {img2.name}: {len(matches)} raw matches")
    
    print("\n=== GEOMETRIC VERIFICATION ===")
    # Check two-view geometries
    for img1_id, img1 in images.items():
        for img2_id, img2 in images.items():
            if img1_id >= img2_id:
                continue
                
            try:
                geom = db.read_two_view_geometry(img1_id, img2_id)
                inliers = np.sum(geom.inlier_matches)
                total = len(geom.matches)
                ratio = inliers / max(1, total)
                print(f"{img1.name} <-> {img2.name}: {inliers}/{total} inliers ({ratio:.1%})")
            except:
                pass
    
    # Load reconstruction
    print("\n=== RECONSTRUCTION ANALYSIS ===")
    for recon_idx in [0, 1]:
        sparse_dir = f"output/sparse/{recon_idx}"
        if not Path(sparse_dir).exists():
            continue
            
        print(f"\nReconstruction {recon_idx}:")
        reconstruction = pycolmap.Reconstruction(str(sparse_dir))
        
        # Count shared 3D points between image pairs
        image_pairs = {}
        for point3d in reconstruction.points3D.values():
            observing_images = [(elem.image_id, elem.point2D_idx) for elem in point3d.track.elements]
            
            for i in range(len(observing_images)):
                for j in range(i + 1, len(observing_images)):
                    img1_id = observing_images[i][0]
                    img2_id = observing_images[j][0]
                    
                    if img1_id not in reconstruction.images or img2_id not in reconstruction.images:
                        continue
                        
                    pair_key = (reconstruction.images[img1_id].name, reconstruction.images[img2_id].name)
                    if pair_key not in image_pairs:
                        image_pairs[pair_key] = 0
                    image_pairs[pair_key] += 1
        
        # Report findings
        for (img1_name, img2_name), shared_points in sorted(image_pairs.items(), key=lambda x: x[1], reverse=True):
            raw_count = raw_matches.get((img1_name, img2_name), 0) or raw_matches.get((img2_name, img1_name), 0)
            survival_rate = shared_points / max(1, raw_count)
            print(f"  {img1_name} <-> {img2_name}: {shared_points} 3D points (from {raw_count} raw matches, {survival_rate:.1%} survival)")

if __name__ == "__main__":
    analyze_matches()