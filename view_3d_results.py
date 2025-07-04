#!/usr/bin/env python3
"""
View 3D reconstruction results from COLMAP output
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import pycolmap
import argparse


def visualize_colmap_reconstruction(model_path):
    """Load and visualize COLMAP reconstruction"""
    
    # Load the reconstruction
    reconstruction = pycolmap.Reconstruction(model_path)
    
    print(f"\nReconstruction Statistics:")
    print(f"Number of images: {len(reconstruction.images)}")
    print(f"Number of 3D points: {len(reconstruction.points3D)}")
    print(f"Number of cameras: {len(reconstruction.cameras)}")
    
    # Extract 3D points
    points = []
    colors = []
    
    for point3D_id, point3D in reconstruction.points3D.items():
        points.append(point3D.xyz)
        colors.append(point3D.color / 255.0)  # Normalize to [0, 1]
    
    points = np.array(points)
    colors = np.array(colors)
    
    # Create 3D visualization
    fig = plt.figure(figsize=(15, 10))
    
    # 3D scatter plot
    ax = fig.add_subplot(121, projection='3d')
    
    # Subsample points if too many
    if len(points) > 10000:
        indices = np.random.choice(len(points), 10000, replace=False)
        points_vis = points[indices]
        colors_vis = colors[indices]
    else:
        points_vis = points
        colors_vis = colors
    
    ax.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2], 
               c=colors_vis, s=1, alpha=0.5)
    
    # Add camera positions
    camera_positions = []
    for image_id, image in reconstruction.images.items():
        # Get rotation matrix from quaternion
        R = pycolmap.qvec_to_rotmat(image.qvec)
        t = image.tvec
        # Camera position in world coordinates
        camera_pos = -R.T @ t
        camera_positions.append(camera_pos)
    
    camera_positions = np.array(camera_positions)
    ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
               c='red', s=100, marker='^', label='Cameras')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Reconstruction')
    ax.legend()
    
    # Top-down view
    ax2 = fig.add_subplot(122)
    ax2.scatter(points[:, 0], points[:, 1], c=colors, s=1, alpha=0.5)
    ax2.scatter(camera_positions[:, 0], camera_positions[:, 1], 
                c='red', s=100, marker='^', label='Cameras')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Top-down View')
    ax2.axis('equal')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('reconstruction_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print camera information
    print("\nCamera Information:")
    for idx, (image_id, image) in enumerate(reconstruction.images.items()):
        print(f"  Image {idx + 1}: {image.name}")
        camera = reconstruction.cameras[image.camera_id]
        print(f"    Camera model: {camera.model_name}")
        print(f"    Focal length: {camera.focal_length}")


def export_to_ply(model_path, output_path):
    """Export reconstruction to PLY format for viewing in MeshLab or CloudCompare"""
    
    reconstruction = pycolmap.Reconstruction(model_path)
    
    # Extract points
    points = []
    colors = []
    
    for point3D_id, point3D in reconstruction.points3D.items():
        points.append(point3D.xyz)
        colors.append(point3D.color)
    
    points = np.array(points)
    colors = np.array(colors)
    
    # Write PLY file
    with open(output_path, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write points
        for point, color in zip(points, colors):
            f.write(f"{point[0]} {point[1]} {point[2]} ")
            f.write(f"{int(color[0])} {int(color[1])} {int(color[2])}\n")
    
    print(f"Exported {len(points)} points to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View 3D reconstruction results")
    parser.add_argument("--model-path", type=str, default="output_multires/sparse/0",
                        help="Path to COLMAP reconstruction")
    parser.add_argument("--export-ply", action="store_true",
                        help="Export to PLY format")
    args = parser.parse_args()
    
    # Visualize
    visualize_colmap_reconstruction(args.model_path)
    
    # Export to PLY if requested
    if args.export_ply:
        ply_path = "reconstruction.ply"
        export_to_ply(args.model_path, ply_path)
        print(f"\nYou can view the PLY file in:")
        print("- MeshLab: https://www.meshlab.net/")
        print("- CloudCompare: https://www.cloudcompare.org/")
        print("- Online viewer: https://www.3dviewer.net/")