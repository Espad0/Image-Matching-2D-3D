"""Export COLMAP reconstruction to PLY file for viewing."""
import pycolmap
import numpy as np
from pathlib import Path

# Load the reconstruction
reconstruction_path = Path("output/sparse/0")
reconstruction = pycolmap.Reconstruction(reconstruction_path)

# Export to PLY
output_file = "output/point_cloud.ply"

# Write PLY header
with open(output_file, 'w') as f:
    # Get points
    points = []
    colors = []
    for point3D_id, point3D in reconstruction.points3D.items():
        points.append(point3D.xyz)
        colors.append(point3D.color)
    
    points = np.array(points)
    colors = np.array(colors, dtype=np.uint8)
    
    # Write PLY header
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
        f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")

print(f"Exported {len(points)} points to {output_file}")
print("\nTo view the point cloud:")
print("1. Install MeshLab: brew install --cask meshlab")
print("2. Open MeshLab and load output/point_cloud.ply")
print("\nOr use CloudCompare: brew install --cask cloudcompare")