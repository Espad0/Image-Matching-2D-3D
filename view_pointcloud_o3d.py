#!/usr/bin/env python3
"""
View binary point cloud files using Open3D
"""

import numpy as np
import open3d as o3d
import argparse
from pathlib import Path


def load_binary_pointcloud(file_path, dtype=np.float32):
    """Load point cloud from binary file."""
    # Load binary file
    points = np.fromfile(file_path, dtype=dtype)
    
    # Try to reshape to (N, 3) for x,y,z format
    if len(points) % 3 != 0:
        raise ValueError(f"Number of values ({len(points)}) is not divisible by 3")
    
    points = points.reshape(-1, 3)
    print(f"Loaded {len(points)} points from {file_path}")
    
    return points


def create_point_cloud(points, colors=None):
    """Create Open3D point cloud object."""
    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Add colors if provided
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Use height-based coloring as default
        heights = points[:, 2]  # Z-axis
        normalized_heights = (heights - heights.min()) / (heights.max() - heights.min())
        colors = np.zeros((len(points), 3))
        colors[:, 0] = normalized_heights  # Red channel varies with height
        colors[:, 2] = 1 - normalized_heights  # Blue channel inverse
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


def visualize_pointcloud(pcd, window_name="Point Cloud Viewer", input_path=None):
    """Visualize point cloud using Open3D."""
    # Print controls
    print("\nVisualization Controls:")
    print("  - Mouse: Rotate view")
    print("  - Scroll: Zoom in/out")
    print("  - R: Reset view")
    print("  - Q/Esc: Quit")
    print("  - H: Print help")
    print("  - P: Take screenshot")
    
    try:
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        if not vis.create_window(window_name=window_name, width=1024, height=768):
            raise RuntimeError("Failed to create visualization window")
        
        vis.add_geometry(pcd)
        
        # Set rendering options
        render_option = vis.get_render_option()
        if render_option is not None:
            render_option.point_size = 2.0
            render_option.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
        
        # Run visualizer
        vis.run()
        vis.destroy_window()
        
    except Exception as e:
        print(f"\nError: Failed to create visualization window: {e}")
        print("\nPossible solutions:")
        print("1. If using SSH, enable X11 forwarding: ssh -X user@host")
        print("2. Set DISPLAY environment variable: export DISPLAY=:0")
        print("3. Install required OpenGL libraries:")
        print("   sudo apt-get install libgl1-mesa-glx libglu1-mesa")
        print("4. Use headless rendering or save to file instead")
        
        # Offer alternative: save to file
        if input_path:
            response = input("\nWould you like to save the point cloud to a PLY file instead? (y/n): ")
            if response.lower() == 'y':
                output_path = input_path.with_suffix('.ply')
                print(f"\nSaving point cloud to {output_path}...")
                o3d.io.write_point_cloud(str(output_path), pcd)
                print(f"Point cloud saved successfully to {output_path}")
                print("You can view it later with: meshlab or cloudcompare")


def main():
    parser = argparse.ArgumentParser(description="View binary point cloud files using Open3D")
    parser.add_argument("input", type=str, help="Path to binary point cloud file")
    parser.add_argument("--dtype", type=str, default="float32",
                       choices=["float32", "float64", "int16", "int32"],
                       help="Data type of the binary file")
    parser.add_argument("--downsample", type=float, default=None,
                       help="Voxel size for downsampling (optional)")
    parser.add_argument("--estimate-normals", action="store_true",
                       help="Estimate point normals for better visualization")
    parser.add_argument("--remove-outliers", action="store_true",
                       help="Remove statistical outliers")
    args = parser.parse_args()
    
    # Check if file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found!")
        return
    
    # Map string dtype to numpy dtype
    dtype_map = {
        "float32": np.float32,
        "float64": np.float64,
        "int16": np.int16,
        "int32": np.int32
    }
    dtype = dtype_map[args.dtype]
    
    # Load point cloud
    print(f"Loading point cloud from {input_path}...")
    try:
        points = load_binary_pointcloud(input_path, dtype)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Create Open3D point cloud
    pcd = create_point_cloud(points)
    
    # Print statistics
    print(f"\nPoint cloud statistics:")
    print(f"  Total points: {len(pcd.points)}")
    bounds = pcd.get_axis_aligned_bounding_box()
    print(f"  Bounding box: {bounds}")
    center = pcd.get_center()
    print(f"  Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    
    # Optional: Remove outliers
    if args.remove_outliers:
        print("\nRemoving outliers...")
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"  Points after outlier removal: {len(pcd.points)}")
    
    # Optional: Downsample
    if args.downsample:
        print(f"\nDownsampling with voxel size {args.downsample}...")
        pcd = pcd.voxel_down_sample(voxel_size=args.downsample)
        print(f"  Points after downsampling: {len(pcd.points)}")
    
    # Optional: Estimate normals
    if args.estimate_normals:
        print("\nEstimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
    
    # Visualize
    print("\nStarting visualization...")
    visualize_pointcloud(pcd, window_name=f"Point Cloud: {input_path.name}", input_path=input_path)


if __name__ == "__main__":
    main()