#!/usr/bin/env python3
"""
View 3D reconstruction from COLMAP using matplotlib
"""

import numpy as np
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack bytes from file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3D_binary(path_to_model_file):
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
            track_elems = read_next_bytes(fid, num_bytes=8*track_length,
                                         format_char_sequence="ii"*track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = {"xyz": xyz, "rgb": rgb, "error": error,
                                   "image_ids": image_ids, "point2D_idxs": point2D_idxs}
    return points3D


def save_point_cloud(points3D, output_path):
    """Save point cloud to numpy file."""
    xyz_list = []
    rgb_list = []
    for point_id, point_data in points3D.items():
        xyz_list.append(point_data["xyz"])
        rgb_list.append(point_data["rgb"])
    
    xyz_array = np.array(xyz_list)
    rgb_array = np.array(rgb_list)
    
    np.savez(output_path, points=xyz_array, colors=rgb_array)
    print(f"Saved {len(xyz_array)} points to {output_path}")
    return xyz_array, rgb_array


def visualize_point_cloud(points3D, sample_rate=1, point_size=1):
    """Visualize point cloud using matplotlib."""
    # Extract points and colors
    xyz_list = []
    rgb_list = []
    for i, (point_id, point_data) in enumerate(points3D.items()):
        if i % sample_rate == 0:  # Subsample for visualization
            xyz_list.append(point_data["xyz"])
            rgb_list.append(point_data["rgb"] / 255.0)  # Normalize to [0, 1]
    
    xyz_array = np.array(xyz_list)
    rgb_array = np.array(rgb_list)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    
    # Plot points
    scatter = ax.scatter(xyz_array[:, 0], xyz_array[:, 1], xyz_array[:, 2],
                        c=rgb_array, s=point_size, alpha=0.6)
    
    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3D Reconstruction ({len(xyz_array)} points shown)")
    
    # Set equal aspect ratio
    max_range = np.array([xyz_array[:, 0].max()-xyz_array[:, 0].min(),
                         xyz_array[:, 1].max()-xyz_array[:, 1].min(),
                         xyz_array[:, 2].max()-xyz_array[:, 2].min()]).max() / 2.0
    
    mid_x = (xyz_array[:, 0].max()+xyz_array[:, 0].min()) * 0.5
    mid_y = (xyz_array[:, 1].max()+xyz_array[:, 1].min()) * 0.5
    mid_z = (xyz_array[:, 2].max()+xyz_array[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add controls info
    ax.text2D(0.05, 0.95, "Use mouse to rotate view", transform=ax.transAxes)
    
    return fig, ax


def main():
    parser = argparse.ArgumentParser(description="View COLMAP 3D reconstruction")
    parser.add_argument("--input", type=str, default="featureout/colmap_reconstruction/0/points3D.bin",
                       help="Path to points3D.bin file")
    parser.add_argument("--output", type=str, default="featureout/reconstruction.npz",
                       help="Output path for saved point cloud")
    parser.add_argument("--sample-rate", type=int, default=1,
                       help="Subsample rate for visualization (1 = show all points)")
    parser.add_argument("--point-size", type=float, default=1,
                       help="Size of points in visualization")
    args = parser.parse_args()
    
    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found!")
        print("\nAvailable reconstructions:")
        for sparse_dir in Path(".").rglob("sparse/*/points3D.bin"):
            print(f"  {sparse_dir}")
        return
    
    # Read point cloud
    print(f"Reading point cloud from {input_path}...")
    points3D = read_points3D_binary(input_path)
    print(f"Loaded {len(points3D)} 3D points")
    
    # Save point cloud
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xyz_array, rgb_array = save_point_cloud(points3D, args.output)
    
    # Print statistics
    print(f"\nPoint cloud statistics:")
    print(f"  X range: [{xyz_array[:, 0].min():.3f}, {xyz_array[:, 0].max():.3f}]")
    print(f"  Y range: [{xyz_array[:, 1].min():.3f}, {xyz_array[:, 1].max():.3f}]")
    print(f"  Z range: [{xyz_array[:, 2].min():.3f}, {xyz_array[:, 2].max():.3f}]")
    
    # Visualize
    print(f"\nVisualizing with sample rate {args.sample_rate}...")
    fig, ax = visualize_point_cloud(points3D, args.sample_rate, args.point_size)
    
    plt.show()


if __name__ == "__main__":
    main()
