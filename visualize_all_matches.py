import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_matches(img1_path, img2_path, kpts1, kpts2, matches, ax, max_matches=100):
    """Visualize matches between two images on given axis"""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # Convert BGR to RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Stack images horizontally
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img_combined = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    img_combined[:h1, :w1] = img1
    img_combined[:h2, w1:] = img2
    
    ax.imshow(img_combined)
    
    # Draw matches
    if len(matches) > max_matches:
        # Random sample matches for visualization
        np.random.seed(42)  # For consistent visualization
        indices = np.random.choice(len(matches), max_matches, replace=False)
        matches_vis = matches[indices]
    else:
        matches_vis = matches
    
    # Use consistent colors based on match quality
    for i, match in enumerate(matches_vis):
        idx1, idx2 = match
        pt1 = kpts1[idx1]
        pt2 = kpts2[idx2]
        
        # Color gradient from red to green based on match index
        color = plt.cm.RdYlGn(i / len(matches_vis))[:3]
        
        # Draw match line
        ax.plot([pt1[0], pt2[0] + w1], [pt1[1], pt2[1]], 
                color=color, linewidth=1, alpha=0.7)
    
    # Add title with match count
    title = f"{os.path.basename(img1_path)} - {os.path.basename(img2_path)}\n{len(matches)} matches"
    ax.set_title(title, fontsize=10)
    ax.axis('off')

# Load the results
feature_dir = './featureout'
img_dir = 'examples/images'

# Read keypoints
keypoints = {}
with h5py.File(f'{feature_dir}/keypoints.h5', 'r') as f:
    for key in f.keys():
        keypoints[key] = f[key][:]

# Count total pairs
match_pairs = []
with h5py.File(f'{feature_dir}/matches.h5', 'r') as f:
    for img1 in f.keys():
        for img2 in f[img1].keys():
            match_pairs.append((img1, img2))

# Create figure with subplots for all pairs
n_pairs = len(match_pairs)
cols = 2  # 2 columns
rows = (n_pairs + cols - 1) // cols  # Calculate rows needed

fig, axes = plt.subplots(rows, cols, figsize=(20, 10 * rows))
if rows == 1:
    axes = axes.reshape(1, -1)
if n_pairs == 1:
    axes = [[axes]]

# Visualize each pair
with h5py.File(f'{feature_dir}/matches.h5', 'r') as f:
    for idx, (img1_key, img2_key) in enumerate(match_pairs):
        row = idx // cols
        col = idx % cols
        
        matches = f[img1_key][img2_key][:]
        
        img1_path = os.path.join(img_dir, img1_key)
        img2_path = os.path.join(img_dir, img2_key)
        
        plot_matches(img1_path, img2_path, 
                    keypoints[img1_key], keypoints[img2_key], 
                    matches, axes[row, col], max_matches=150)

# Hide empty subplots if any
for idx in range(n_pairs, rows * cols):
    row = idx // cols
    col = idx % cols
    axes[row, col].axis('off')

plt.suptitle(f'All Match Pairs from kaggle-py.py ({n_pairs} pairs total)', fontsize=16)
plt.tight_layout()
plt.savefig('kaggle_all_matches_visualization.png', dpi=150, bbox_inches='tight')
print(f"Saved visualization of all {n_pairs} match pairs to kaggle_all_matches_visualization.png")

# Also create individual high-res visualizations
os.makedirs('kaggle_matches_individual', exist_ok=True)

with h5py.File(f'{feature_dir}/matches.h5', 'r') as f:
    for img1_key, img2_key in match_pairs:
        matches = f[img1_key][img2_key][:]
        
        fig_single, ax_single = plt.subplots(1, 1, figsize=(20, 10))
        
        img1_path = os.path.join(img_dir, img1_key)
        img2_path = os.path.join(img_dir, img2_key)
        
        plot_matches(img1_path, img2_path, 
                    keypoints[img1_key], keypoints[img2_key], 
                    matches, ax_single, max_matches=500)
        
        output_name = f"{img1_key.split('.')[0]}_{img2_key.split('.')[0]}_matches.png"
        plt.savefig(f'kaggle_matches_individual/{output_name}', dpi=200, bbox_inches='tight')
        plt.close(fig_single)
        print(f"Saved individual visualization: {output_name}")

print("\nMatch statistics:")
with h5py.File(f'{feature_dir}/matches.h5', 'r') as f:
    for img1_key, img2_key in match_pairs:
        n_matches = len(f[img1_key][img2_key])
        print(f"{img1_key} -> {img2_key}: {n_matches} matches")