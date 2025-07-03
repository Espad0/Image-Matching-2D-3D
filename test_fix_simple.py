"""Simple test to verify geometric verification fix."""
import pycolmap
import sqlite3
import numpy as np

# Create a test database with some dummy matches
db_path = "test_colmap.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create tables
cursor.execute("""CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY,
    model INTEGER,
    width INTEGER,
    height INTEGER,
    params BLOB,
    prior_focal_length INTEGER
)""")

cursor.execute("""CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY,
    name TEXT,
    camera_id INTEGER,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL
)""")

cursor.execute("""CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY,
    rows INTEGER,
    cols INTEGER,
    data BLOB
)""")

cursor.execute("""CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY,
    rows INTEGER,
    cols INTEGER,
    data BLOB
)""")

cursor.execute("""CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY,
    rows INTEGER,
    cols INTEGER,
    data BLOB
)""")

cursor.execute("""CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY,
    rows INTEGER,
    cols INTEGER,
    data BLOB,
    config INTEGER,
    F BLOB,
    E BLOB,
    H BLOB
)""")

# Add dummy data
# Camera
params = np.array([1000.0, 640.0, 480.0, 0.1], dtype=np.float64)
cursor.execute("INSERT INTO cameras VALUES (1, 2, 1280, 960, ?, 0)", 
               (params.tobytes(),))

# Images
cursor.execute("INSERT INTO images VALUES (1, 'img1.jpg', 1, 1, 0, 0, 0, 0, 0, 0)")
cursor.execute("INSERT INTO images VALUES (2, 'img2.jpg', 1, 1, 0, 0, 0, 0, 0, 0)")

# Keypoints (100 random points)
kpts = np.random.rand(100, 2).astype(np.float32) * 1000
cursor.execute("INSERT INTO keypoints VALUES (1, 100, 2, ?)", (kpts.tobytes(),))
cursor.execute("INSERT INTO keypoints VALUES (2, 100, 2, ?)", (kpts.tobytes(),))

# Descriptors (dummy)
desc = np.zeros((100, 128), dtype=np.uint8)
cursor.execute("INSERT INTO descriptors VALUES (1, 100, 128, ?)", (desc.tobytes(),))
cursor.execute("INSERT INTO descriptors VALUES (2, 100, 128, ?)", (desc.tobytes(),))

# Matches
matches = np.column_stack([np.arange(50), np.arange(50)]).astype(np.uint32)
pair_id = 1 * (2**31 - 1) + 2
cursor.execute("INSERT INTO matches VALUES (?, 50, 2, ?)", (pair_id, matches.tobytes()))

conn.commit()

# Check before verification
cursor.execute("SELECT COUNT(*) FROM two_view_geometries")
before_count = cursor.fetchone()[0]
print(f"Two-view geometries before verification: {before_count}")

conn.close()

# Run geometric verification
print("\nRunning pycolmap.match_exhaustive...")
try:
    pycolmap.match_exhaustive(db_path)
    print("✅ Geometric verification completed successfully!")
except Exception as e:
    print(f"❌ Geometric verification failed: {e}")

# Check after verification
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM two_view_geometries")
after_count = cursor.fetchone()[0]
print(f"\nTwo-view geometries after verification: {after_count}")

if after_count > before_count:
    print("\n✅ SUCCESS: The fix is working! Two-view geometries were added.")
else:
    print("\n❌ FAILED: No two-view geometries were added.")

conn.close()

# Clean up
import os
os.remove(db_path)