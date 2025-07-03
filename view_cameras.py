"""Visualize camera positions and orientations."""
import pycolmap
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load reconstruction
reconstruction = pycolmap.Reconstruction("output/sparse/0")

# Extract camera positions
camera_positions = []
camera_names = []
camera_orientations = []

for img_id, image in reconstruction.images.items():
    # Get camera position (projection center)
    if hasattr(image, 'projection_center'):
        pos = image.projection_center()
    else:
        # Calculate from transformation
        R = image.cam_from_world.rotation.matrix()
        t = image.cam_from_world.translation
        pos = -R.T @ t
    
    camera_positions.append(pos)
    camera_names.append(image.name)
    
    # Get viewing direction
    if hasattr(image, 'viewing_direction'):
        direction = image.viewing_direction()
    else:
        # Camera looks down the z-axis in camera coordinates
        direction = R.T @ np.array([0, 0, 1])
    camera_orientations.append(direction)

camera_positions = np.array(camera_positions)
camera_orientations = np.array(camera_orientations)

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot camera positions
ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
           c='red', s=100, marker='o', label='Cameras')

# Plot camera orientations as arrows
for i, (pos, direction, name) in enumerate(zip(camera_positions, camera_orientations, camera_names)):
    # Draw arrow showing viewing direction
    ax.quiver(pos[0], pos[1], pos[2], 
              direction[0], direction[1], direction[2], 
              length=0.5, color='blue', alpha=0.6)
    
    # Add camera label
    ax.text(pos[0], pos[1], pos[2], f'  {i+1}', fontsize=8)

# Also plot a sample of 3D points for context
points_sample = []
for i, (pt_id, point3D) in enumerate(reconstruction.points3D.items()):
    if i % 10 == 0:  # Sample every 10th point
        points_sample.append(point3D.xyz)
    if len(points_sample) > 1000:  # Limit to 1000 points
        break

if points_sample:
    points_sample = np.array(points_sample)
    ax.scatter(points_sample[:, 0], points_sample[:, 1], points_sample[:, 2], 
               c='gray', s=1, alpha=0.3, label='3D Points (sample)')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Camera Positions and Orientations')
ax.legend()

# Print camera information
print("Camera Information:")
for i, (name, pos) in enumerate(zip(camera_names, camera_positions)):
    print(f"{i+1}. {name}: position = [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

plt.show()