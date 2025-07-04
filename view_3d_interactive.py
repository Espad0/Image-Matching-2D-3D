#!/usr/bin/env python3
"""
Interactive 3D viewer for COLMAP reconstruction using matplotlib
"""

import numpy as np
import struct
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse


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


def interactive_3d_viewer(model_path, max_points=10000):
    """Create an interactive 3D viewer using matplotlib."""
    
    # Read data
    points3D = read_points3d_binary(Path(model_path) / "points3D.bin")
    images = read_images_binary(Path(model_path) / "images.bin")
    
    print(f"\nReconstruction loaded:")
    print(f"  - {len(points3D)} 3D points")
    print(f"  - {len(images)} cameras")
    
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
    
    # Subsample points if too many for performance
    if len(points) > max_points:
        print(f"Subsampling to {max_points} points for performance...")
        indices = np.random.choice(len(points), max_points, replace=False)
        points_vis = points[indices]
        colors_vis = colors[indices]
    else:
        points_vis = points
        colors_vis = colors
    
    # Create interactive 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    scatter = ax.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2],
                        c=colors_vis, s=1, alpha=0.6, edgecolors='none')
    
    # Plot cameras
    camera_scatter = ax.scatter(camera_positions[:, 0], 
                               camera_positions[:, 1], 
                               camera_positions[:, 2],
                               c='red', s=100, marker='^', 
                               edgecolors='black', linewidth=2, 
                               label='Cameras', alpha=0.9)
    
    # Add camera labels
    for i, (pos, name) in enumerate(zip(camera_positions, camera_names)):
        ax.text(pos[0], pos[1], pos[2], f' {i+1}', fontsize=10, color='red')
    
    # Set labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f'3D Reconstruction\n{len(points)} points, {len(images)} cameras', 
                 fontsize=14, pad=20)
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Set initial view angle
    ax.view_init(elev=20, azim=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Make the plot look better
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    
    # Print instructions
    print("\nInteractive 3D Viewer Controls:")
    print("  - Left mouse drag: Rotate view")
    print("  - Right mouse drag: Zoom")
    print("  - Middle mouse drag: Pan")
    print("  - 'q': Quit")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  - Mean reprojection error: {np.mean(errors):.2f} pixels")
    print(f"  - Point cloud bounds:")
    print(f"    X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"    Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"    Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    plt.tight_layout()
    plt.show()


def create_multi_view_plot(model_path, max_points=20000):
    """Create a figure with multiple viewing angles."""
    
    # Read data
    points3D = read_points3d_binary(Path(model_path) / "points3D.bin")
    images = read_images_binary(Path(model_path) / "images.bin")
    
    # Extract points and cameras
    points = []
    colors = []
    for point_id, point_data in points3D.items():
        points.append(point_data["xyz"])
        colors.append(point_data["rgb"] / 255.0)
    
    points = np.array(points)
    colors = np.array(colors)
    
    camera_positions = []
    for img_id, img_data in images.items():
        R = qvec2rotmat(img_data["qvec"])
        t = img_data["tvec"]
        camera_pos = -R.T @ t
        camera_positions.append(camera_pos)
    
    camera_positions = np.array(camera_positions)
    
    # Subsample if needed
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        colors = colors[indices]
    
    # Create figure with multiple views
    fig = plt.figure(figsize=(16, 8))
    
    # View 1: Default perspective
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                c=colors, s=0.5, alpha=0.5)
    ax1.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
                c='red', s=100, marker='^', edgecolors='black', linewidth=1)
    ax1.set_title('Perspective View')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=20, azim=45)
    
    # View 2: Top-down view
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2],
                c=colors, s=0.5, alpha=0.5)
    ax2.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
                c='red', s=100, marker='^', edgecolors='black', linewidth=1)
    ax2.set_title('Top-Down View')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(elev=90, azim=0)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive 3D viewer for COLMAP reconstruction")
    parser.add_argument("--model-path", type=str, default="output_imc2023/sparse/0",
                        help="Path to COLMAP model directory")
    parser.add_argument("--max-points", type=int, default=10000,
                        help="Maximum number of points to display (for performance)")
    parser.add_argument("--multi-view", action="store_true",
                        help="Show multiple views instead of interactive")
    args = parser.parse_args()
    
    if args.multi_view:
        create_multi_view_plot(args.model_path, args.max_points)
    else:
        interactive_3d_viewer(args.model_path, args.max_points)