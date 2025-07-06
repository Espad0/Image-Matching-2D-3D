import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_matches(img1_path, img2_path, kpts1, kpts2, matches, max_matches=100):
    """Visualize matches between two images"""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # Convert BGR to RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    
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
        indices = np.random.choice(len(matches), max_matches, replace=False)
        matches_vis = matches[indices]
    else:
        matches_vis = matches
    
    for match in matches_vis:
        idx1, idx2 = match
        pt1 = kpts1[idx1]
        pt2 = kpts2[idx2]
        
        # Draw match line
        ax.plot([pt1[0], pt2[0] + w1], [pt1[1], pt2[1]], 
                color=np.random.rand(3,), linewidth=1, alpha=0.7)
    
    ax.set_title(f"{os.path.basename(img1_path)} - {os.path.basename(img2_path)}: {len(matches)} matches")
    ax.axis('off')
    
    return fig

# Load the results
feature_dir = './featureout'
img_dir = 'examples/images'

# Read keypoints
keypoints = {}
with h5py.File(f'{feature_dir}/keypoints.h5', 'r') as f:
    for key in f.keys():
        keypoints[key] = f[key][:]
        print(f"Image {key}: {len(keypoints[key])} keypoints")

# Read matches and visualize one pair
with h5py.File(f'{feature_dir}/matches.h5', 'r') as f:
    # Get first image pair
    img1_key = list(f.keys())[0]
    img2_key = list(f[img1_key].keys())[0]
    
    matches = f[img1_key][img2_key][:]
    
    print(f"\nVisualizing matches between {img1_key} and {img2_key}")
    print(f"Number of matches: {len(matches)}")
    
    # Create visualization
    img1_path = os.path.join(img_dir, img1_key)
    img2_path = os.path.join(img_dir, img2_key)
    
    fig = plot_matches(img1_path, img2_path, 
                      keypoints[img1_key], keypoints[img2_key], 
                      matches, max_matches=200)
    
    plt.tight_layout()
    plt.savefig('kaggle_matches_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

# Print summary statistics
print("\n=== Summary ===")
print(f"Total images: {len(keypoints)}")
total_matches = 0
match_pairs = 0
with h5py.File(f'{feature_dir}/matches.h5', 'r') as f:
    for img1 in f.keys():
        for img2 in f[img1].keys():
            n_matches = len(f[img1][img2])
            total_matches += n_matches
            match_pairs += 1
            print(f"{img1} -> {img2}: {n_matches} matches")

print(f"\nTotal image pairs with matches: {match_pairs}")
print(f"Average matches per pair: {total_matches / match_pairs:.1f}")