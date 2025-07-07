#!/usr/bin/env python3
"""
Convert COLMAP points3D.bin to simple binary format for Open3D visualization
"""

import numpy as np
import struct
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


def save_binary_pointcloud(points3D, output_path, include_colors=True):
    """Save point cloud as simple binary file."""
    xyz_list = []
    rgb_list = []
    
    for point_id, point_data in points3D.items():
        xyz_list.append(point_data["xyz"])
        rgb_list.append(point_data["rgb"] / 255.0)  # Normalize to [0, 1]
    
    xyz_array = np.array(xyz_list, dtype=np.float32)
    rgb_array = np.array(rgb_list, dtype=np.float32)
    
    if include_colors:
        # Interleave XYZ and RGB data: x, y, z, r, g, b, x, y, z, r, g, b, ...
        combined = np.empty((len(xyz_array), 6), dtype=np.float32)
        combined[:, 0:3] = xyz_array
        combined[:, 3:6] = rgb_array
        combined.tofile(output_path)
        print(f"Saved {len(xyz_array)} points with colors (6 values per point) to {output_path}")
    else:
        # Save only XYZ data
        xyz_array.tofile(output_path)
        print(f"Saved {len(xyz_array)} points (3 values per point) to {output_path}")
    
    return len(xyz_array)


def main():
    parser = argparse.ArgumentParser(description="Convert COLMAP points3D.bin to simple binary format")
    parser.add_argument("input", type=str, help="Path to COLMAP points3D.bin file")
    parser.add_argument("output", type=str, help="Output binary file path")
    parser.add_argument("--no-colors", action="store_true", help="Save only XYZ coordinates without colors")
    args = parser.parse_args()
    
    # Check if input exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found!")
        return
    
    # Read COLMAP data
    print(f"Reading COLMAP data from {input_path}...")
    points3D = read_points3D_binary(input_path)
    print(f"Found {len(points3D)} points")
    
    # Save as binary
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    num_points = save_binary_pointcloud(points3D, output_path, include_colors=not args.no_colors)
    
    # Print info for visualization
    if args.no_colors:
        print(f"\nTo visualize with Open3D, use:")
        print(f"  python view_pointcloud_o3d.py {output_path}")
    else:
        print(f"\nNote: The binary file contains interleaved XYZ-RGB data (6 float32 values per point)")
        print(f"You may need to modify the visualization script to handle the color data")


if __name__ == "__main__":
    main()