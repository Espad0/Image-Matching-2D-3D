import pycolmap
from pathlib import Path

# Read the reconstruction
sparse_path = Path("output/sparse/0")
reconstruction = pycolmap.Reconstruction(str(sparse_path))

print(f"Number of registered images: {len(reconstruction.images)}")
print(f"Number of 3D points: {len(reconstruction.points3D)}")
print(f"Number of cameras: {len(reconstruction.cameras)}")

# List registered images
print("\nRegistered images:")
for img_id, image in reconstruction.images.items():
    print(f"  - {image.name} (ID: {img_id})")

# Show camera info
print("\nCamera models:")
for cam_id, camera in reconstruction.cameras.items():
    print(f"  - Camera {cam_id}: {camera.model_name}")