"""Simple 3D viewer for the reconstruction using matplotlib."""
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# Load reconstruction data
reconstruction_file = Path("output/reconstruction.json")

if reconstruction_file.exists():
    # Load from JSON export
    with open(reconstruction_file, 'r') as f:
        data = json.load(f)
    
    # Extract 3D points
    points = []
    colors = []
    for pt_id, point_data in data['points3D'].items():
        points.append(point_data['xyz'])
        colors.append(np.array(point_data['rgb']) / 255.0)
    
    points = np.array(points)
    colors = np.array(colors)
else:
    # Load directly from COLMAP
    import pycolmap
    reconstruction = pycolmap.Reconstruction("output/sparse/0")
    
    points = []
    colors = []
    for pt_id, point3D in reconstruction.points3D.items():
        points.append(point3D.xyz)
        colors.append(point3D.color / 255.0)
    
    points = np.array(points)
    colors = np.array(colors)

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
           c=colors, s=1, alpha=0.6)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'3D Reconstruction ({len(points)} points)')

# Equal aspect ratio
max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                     points[:, 1].max()-points[:, 1].min(),
                     points[:, 2].max()-points[:, 2].min()]).max() / 2.0

mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()

# Print statistics
print(f"\nReconstruction Statistics:")
print(f"Number of 3D points: {len(points)}")
print(f"Bounding box:")
print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")