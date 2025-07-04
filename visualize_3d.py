#!/usr/bin/env python3
"""
Direct visualization of COLMAP reconstruction using matplotlib
"""

import numpy as np
import struct
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack bytes from file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3d_binary(path_to_model_file):
    """Read COLMAP points3D.bin file."""
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = binary_point_line_properties[7]
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(fid, num_bytes=8 * track_length,
                                         format_char_sequence="ii" * track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "image_ids": image_ids,
                "point2D_idxs": point2D_idxs
            }
    return points3D


def read_images_binary(path_to_model_file):
    """Read COLMAP images.bin file."""
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D,
                                      format_char_sequence="ddq" * num_points2D)
            
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                  tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            
            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": image_name,
                "xys": xys,
                "point3D_ids": point3D_ids
            }
    return images


def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix."""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def visualize_reconstruction(model_path):
    """Visualize COLMAP reconstruction."""
    
    # Read data
    points3D = read_points3d_binary(Path(model_path) / "points3D.bin")
    images = read_images_binary(Path(model_path) / "images.bin")
    
    print(f"\nReconstruction Statistics:")
    print(f"Number of 3D points: {len(points3D)}")
    print(f"Number of images: {len(images)}")
    
    # Extract 3D points and colors
    points = []
    colors = []
    errors = []
    
    for point_id, point_data in points3D.items():
        points.append(point_data["xyz"])
        colors.append(point_data["rgb"] / 255.0)
        errors.append(point_data["error"])
    
    points = np.array(points)
    colors = np.array(colors)
    errors = np.array(errors)
    
    # Extract camera positions
    camera_positions = []
    camera_names = []
    
    for img_id, img_data in images.items():
        R = qvec2rotmat(img_data["qvec"])
        t = img_data["tvec"]
        # Camera position in world coordinates
        camera_pos = -R.T @ t
        camera_positions.append(camera_pos)
        camera_names.append(img_data["name"])
    
    camera_positions = np.array(camera_positions)
    
    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    
    # 3D view
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Subsample points for visualization if too many
    max_points = 10000
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points_vis = points[indices]
        colors_vis = colors[indices]
    else:
        points_vis = points
        colors_vis = colors
    
    scatter = ax1.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2],
                         c=colors_vis, s=1, alpha=0.6)
    
    # Add cameras
    ax1.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
               c='red', s=200, marker='^', edgecolors='black', linewidth=2, label='Cameras')
    
    # Add camera labels
    for i, (pos, name) in enumerate(zip(camera_positions, camera_names)):
        ax1.text(pos[0], pos[1], pos[2], f'{i+1}', fontsize=8)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'3D Reconstruction ({len(points)} points)')
    ax1.legend()
    
    # Top view (X-Y)
    ax2 = fig.add_subplot(222)
    ax2.scatter(points[:, 0], points[:, 1], c=colors, s=0.5, alpha=0.5)
    ax2.scatter(camera_positions[:, 0], camera_positions[:, 1],
               c='red', s=200, marker='^', edgecolors='black', linewidth=2)
    for i, (pos, name) in enumerate(zip(camera_positions, camera_names)):
        ax2.text(pos[0], pos[1], f'{i+1}', fontsize=8)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Top View (X-Y)')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    # Side view (X-Z)
    ax3 = fig.add_subplot(223)
    ax3.scatter(points[:, 0], points[:, 2], c=colors, s=0.5, alpha=0.5)
    ax3.scatter(camera_positions[:, 0], camera_positions[:, 2],
               c='red', s=200, marker='^', edgecolors='black', linewidth=2)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('Side View (X-Z)')
    ax3.axis('equal')
    ax3.grid(True, alpha=0.3)
    
    # Error distribution
    ax4 = fig.add_subplot(224)
    ax4.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax4.set_xlabel('Reprojection Error (pixels)')
    ax4.set_ylabel('Number of Points')
    ax4.set_title(f'Error Distribution (mean: {np.mean(errors):.2f} pixels)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print statistics
    print(f"\nMean reprojection error: {np.mean(errors):.2f} pixels")
    print(f"Median reprojection error: {np.median(errors):.2f} pixels")
    print(f"Max reprojection error: {np.max(errors):.2f} pixels")
    
    print("\nCamera names:")
    for i, name in enumerate(camera_names):
        print(f"  Camera {i+1}: {name}")
    
    # Save and show
    output_file = "reconstruction_visualization.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    # Don't show interactively to avoid hanging
    # plt.show()
    
    return points, colors, camera_positions


def export_simple_ply(points, colors, output_path="reconstruction.ply"):
    """Export points to simple PLY format."""
    with open(output_path, 'w') as f:
        # Header
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
        
        # Points
        for point, color in zip(points, colors):
            color_int = (color * 255).astype(int)
            f.write(f"{point[0]} {point[1]} {point[2]} ")
            f.write(f"{color_int[0]} {color_int[1]} {color_int[2]}\n")
    
    print(f"Exported {len(points)} points to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize COLMAP reconstruction")
    parser.add_argument("--model-path", type=str, default="output_multires/sparse/0",
                        help="Path to COLMAP model directory")
    parser.add_argument("--export-ply", action="store_true",
                        help="Export to PLY format")
    args = parser.parse_args()
    
    # Visualize
    points, colors, cameras = visualize_reconstruction(args.model_path)
    
    # Export PLY if requested
    if args.export_ply:
        export_simple_ply(points, colors)
        print("\nYou can view the PLY file in:")
        print("  - MeshLab: https://www.meshlab.net/")
        print("  - CloudCompare: https://www.cloudcompare.org/")
        print("  - Online: https://www.3dviewer.net/")