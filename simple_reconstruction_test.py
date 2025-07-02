"""
Simple test to demonstrate 3D reconstruction with LoFTR matches.
This creates visualizations of intermediate results.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from vision3d.models import LoFTRMatcher

# Output directory
output_dir = Path("simple_reconstruction_output")
output_dir.mkdir(exist_ok=True)

# Test images
img_paths = [
    "examples/images/3DOM_FBK_IMG_1556.png",
    "examples/images/3DOM_FBK_IMG_1552.png"
]

print("Step 1: Loading images and extracting LoFTR matches...")

# Initialize LoFTR
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
matcher = LoFTRMatcher(device=device, config={
    'image_resize': 1024,
    'confidence_threshold': 0.5,  # Higher threshold for better quality
    'use_tta': False
})

# Get matches
mkpts1, mkpts2, mconf = matcher.match_pair(img_paths[0], img_paths[1])
print(f"Found {len(mkpts1)} matches with confidence > 0.5")

# Load images for visualization
img1 = cv2.imread(img_paths[0])
img2 = cv2.imread(img_paths[1])

# Create match visualization
if len(mkpts1) > 0:
    # Create side-by-side visualization
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    vis = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:] = img2
    
    # Draw matches (limit to 200 for clarity)
    indices = np.random.choice(len(mkpts1), min(200, len(mkpts1)), replace=False)
    for idx in indices:
        pt1 = (int(mkpts1[idx, 0]), int(mkpts1[idx, 1]))
        pt2 = (int(mkpts2[idx, 0] + w1), int(mkpts2[idx, 1]))
        
        # Color by confidence
        color = plt.cm.RdYlGn(mconf[idx])[:3]
        color = tuple(int(c * 255) for c in color)
        
        cv2.line(vis, pt1, pt2, color, 1)
        cv2.circle(vis, pt1, 3, color, -1)
        cv2.circle(vis, pt2, 3, color, -1)
    
    # Add text
    cv2.putText(vis, f"{len(mkpts1)} LoFTR matches (showing 200)", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(str(output_dir / "loftr_matches.png"), vis)
    print(f"Saved match visualization to {output_dir / 'loftr_matches.png'}")

print("\nStep 2: Creating a simple triangulation...")

if len(mkpts1) >= 8:
    # Estimate fundamental matrix
    F, mask = cv2.findFundamentalMat(mkpts1, mkpts2, cv2.FM_RANSAC, 1.0, 0.99)
    inliers = mask.ravel() == 1
    print(f"RANSAC found {np.sum(inliers)} inliers out of {len(mkpts1)} matches")
    
    # Get inlier matches
    pts1_inliers = mkpts1[inliers]
    pts2_inliers = mkpts2[inliers]
    
    # Camera intrinsics (approximate)
    h, w = img1.shape[:2]
    focal_length = max(w, h) * 1.2
    K = np.array([
        [focal_length, 0, w/2],
        [0, focal_length, h/2],
        [0, 0, 1]
    ])
    
    # Estimate essential matrix
    E = K.T @ F @ K
    
    # Recover pose
    _, R, t, mask_pose = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)
    print(f"Recovered pose with {np.sum(mask_pose)} valid points")
    
    # Triangulate points
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])
    
    pts1_norm = cv2.undistortPoints(pts1_inliers.reshape(-1, 1, 2), K, None)
    pts2_norm = cv2.undistortPoints(pts2_inliers.reshape(-1, 1, 2), K, None)
    
    points_4d = cv2.triangulatePoints(P1, P2, pts1_norm, pts2_norm)
    points_3d = points_4d[:3] / points_4d[3]
    points_3d = points_3d.T
    
    print(f"Triangulated {len(points_3d)} 3D points")
    
    # Filter points by reprojection error and depth
    valid_points = []
    for i, pt3d in enumerate(points_3d):
        # Check depth
        if pt3d[2] > 0:  # In front of camera
            # Reproject to both images
            pt1_proj = P1 @ np.append(pt3d, 1)
            pt1_proj = pt1_proj[:2] / pt1_proj[2]
            
            pt2_proj = P2 @ np.append(pt3d, 1)
            pt2_proj = pt2_proj[:2] / pt2_proj[2]
            
            # Calculate reprojection error
            err1 = np.linalg.norm(pt1_proj - pts1_inliers[i])
            err2 = np.linalg.norm(pt2_proj - pts2_inliers[i])
            
            if err1 < 5.0 and err2 < 5.0:  # Threshold in pixels
                valid_points.append(pt3d)
    
    valid_points = np.array(valid_points)
    print(f"After filtering: {len(valid_points)} valid 3D points")
    
    # Visualize 3D points
    if len(valid_points) > 0:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points
        ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2], 
                  c='blue', s=1, alpha=0.5)
        
        # Plot cameras
        ax.scatter(0, 0, 0, c='red', s=100, marker='^', label='Camera 1')
        cam2_pos = -R.T @ t
        ax.scatter(cam2_pos[0], cam2_pos[1], cam2_pos[2], 
                  c='green', s=100, marker='^', label='Camera 2')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Simple 3D Reconstruction ({len(valid_points)} points)')
        ax.legend()
        
        # Set equal aspect ratio
        max_range = np.array([
            valid_points[:, 0].max() - valid_points[:, 0].min(),
            valid_points[:, 1].max() - valid_points[:, 1].min(),
            valid_points[:, 2].max() - valid_points[:, 2].min()
        ]).max() / 2.0
        
        mid_x = valid_points[:, 0].mean()
        mid_y = valid_points[:, 1].mean()
        mid_z = valid_points[:, 2].mean()
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.savefig(output_dir / "simple_3d_reconstruction.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved 3D visualization to {output_dir / 'simple_3d_reconstruction.png'}")
        
        # Save point cloud
        np.savetxt(output_dir / "point_cloud.txt", valid_points, 
                   header="X Y Z", comments="")
        print(f"Saved point cloud to {output_dir / 'point_cloud.txt'}")

print(f"\nResults saved to: {output_dir}")
print("\nSummary:")
print(f"- LoFTR found {len(mkpts1)} high-confidence matches")
if len(mkpts1) >= 8:
    print(f"- RANSAC found {np.sum(inliers)} geometric inliers")
    print(f"- Triangulated {len(valid_points)} valid 3D points")
    print(f"- Successfully created a simple 3D reconstruction!")
else:
    print("- Not enough matches for 3D reconstruction")