"""Check COLMAP database contents."""

import sqlite3
import numpy as np

# Connect to database
db = sqlite3.connect('output/colmap.db')
cursor = db.cursor()

# Check cameras
print("=== CAMERAS ===")
cursor.execute("SELECT * FROM cameras")
cameras = cursor.fetchall()
print(f"Number of cameras: {len(cameras)}")
for cam in cameras:
    print(f"Camera {cam[0]}: model={cam[1]}, size={cam[2]}x{cam[3]}")

# Check images
print("\n=== IMAGES ===")
cursor.execute("SELECT * FROM images")
images = cursor.fetchall()
print(f"Number of images: {len(images)}")
for img in images:
    print(f"Image {img[0]}: {img[1]}, camera_id={img[2]}")

# Check keypoints
print("\n=== KEYPOINTS ===")
cursor.execute("SELECT image_id, rows, cols FROM keypoints")
keypoints = cursor.fetchall()
print(f"Number of images with keypoints: {len(keypoints)}")
for kp in keypoints:
    print(f"Image {kp[0]}: {kp[1]} keypoints, dims={kp[2]}")

# Check matches
print("\n=== MATCHES ===")
cursor.execute("SELECT pair_id, rows, cols FROM matches")
matches = cursor.fetchall()
print(f"Number of image pairs with matches: {len(matches)}")
for match in matches:
    print(f"Pair {match[0]}: {match[1]} matches, dims={match[2]}")
    
# Check two_view_geometries
print("\n=== TWO VIEW GEOMETRIES ===")
cursor.execute("SELECT pair_id, rows, cols, config FROM two_view_geometries")
geometries = cursor.fetchall()
print(f"Number of verified pairs: {len(geometries)}")
for geom in geometries:
    print(f"Pair {geom[0]}: {geom[1]} inliers, config={geom[3]}")

db.close()