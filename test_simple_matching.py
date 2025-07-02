"""Simple test to debug LoFTR matching."""

import cv2
import numpy as np
import torch
import kornia.feature as KF
import matplotlib.pyplot as plt

# Load two images
img1_path = "examples/3DOM_FBK_IMG_1516.png"
img2_path = "examples/3DOM_FBK_IMG_1520.png"

# Read images
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

print(f"Image 1 shape: {img1.shape}")
print(f"Image 2 shape: {img2.shape}")

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Resize for LoFTR (needs to be divisible by 8)
target_size = 640
scale1 = target_size / max(gray1.shape)
scale2 = target_size / max(gray2.shape)

new_h1 = int(gray1.shape[0] * scale1)
new_w1 = int(gray1.shape[1] * scale1)
new_h2 = int(gray2.shape[0] * scale2)
new_w2 = int(gray2.shape[1] * scale2)

# Make divisible by 8
new_h1 = (new_h1 // 8) * 8
new_w1 = (new_w1 // 8) * 8
new_h2 = (new_h2 // 8) * 8
new_w2 = (new_w2 // 8) * 8

gray1_resized = cv2.resize(gray1, (new_w1, new_h1))
gray2_resized = cv2.resize(gray2, (new_w2, new_h2))

print(f"Resized image 1: {gray1_resized.shape}")
print(f"Resized image 2: {gray2_resized.shape}")

# Convert to torch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Normalize and add batch and channel dimensions
tensor1 = torch.from_numpy(gray1_resized).float()[None, None] / 255.0
tensor2 = torch.from_numpy(gray2_resized).float()[None, None] / 255.0

tensor1 = tensor1.to(device)
tensor2 = tensor2.to(device)

print(f"Tensor 1 shape: {tensor1.shape}")
print(f"Tensor 2 shape: {tensor2.shape}")

# Initialize LoFTR with pretrained weights
print("Initializing LoFTR...")
try:
    loftr = KF.LoFTR(pretrained='outdoor')
    loftr = loftr.to(device).eval()
    print("LoFTR initialized successfully")
except Exception as e:
    print(f"Error initializing LoFTR: {e}")
    # Try without pretrained weights
    loftr = KF.LoFTR(pretrained=None)
    loftr = loftr.to(device).eval()
    print("LoFTR initialized without pretrained weights")

# Prepare input
input_dict = {
    "image0": tensor1,
    "image1": tensor2
}

# Run LoFTR
print("Running LoFTR matching...")
with torch.no_grad():
    try:
        correspondences = loftr(input_dict)
        print(f"LoFTR output keys: {correspondences.keys()}")
        
        if 'keypoints0' in correspondences:
            mkpts0 = correspondences['keypoints0'].cpu().numpy()
            mkpts1 = correspondences['keypoints1'].cpu().numpy()
            mconf = correspondences['confidence'].cpu().numpy()
            
            print(f"Found {len(mkpts0)} matches")
            if len(mkpts0) > 0:
                print(f"Confidence range: [{mconf.min():.3f}, {mconf.max():.3f}]")
                print(f"Mean confidence: {mconf.mean():.3f}")
                
                # Scale keypoints back to original size
                mkpts0[:, 0] = mkpts0[:, 0] / scale1
                mkpts0[:, 1] = mkpts0[:, 1] / scale1
                mkpts1[:, 0] = mkpts1[:, 0] / scale2
                mkpts1[:, 1] = mkpts1[:, 1] / scale2
                
                # Visualize matches
                fig, ax = plt.subplots(1, 1, figsize=(15, 10))
                
                # Create side-by-side image
                h1, w1 = img1.shape[:2]
                h2, w2 = img2.shape[:2]
                h = max(h1, h2)
                vis = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
                vis[:h1, :w1] = img1
                vis[:h2, w1:] = img2
                
                # Draw matches
                for i in range(min(len(mkpts0), 100)):  # Limit to 100 matches for clarity
                    pt1 = (int(mkpts0[i, 0]), int(mkpts0[i, 1]))
                    pt2 = (int(mkpts1[i, 0] + w1), int(mkpts1[i, 1]))
                    
                    # Color by confidence
                    if mconf[i] > 0.7:
                        color = (0, 255, 0)  # Green
                    elif mconf[i] > 0.4:
                        color = (255, 255, 0)  # Yellow
                    else:
                        color = (255, 0, 0)  # Blue
                    
                    cv2.line(vis, pt1, pt2, color, 1)
                    cv2.circle(vis, pt1, 3, color, -1)
                    cv2.circle(vis, pt2, 3, color, -1)
                
                # Convert BGR to RGB for matplotlib
                vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                ax.imshow(vis_rgb)
                ax.set_title(f"{len(mkpts0)} matches found")
                ax.axis('off')
                
                plt.tight_layout()
                plt.savefig('simple_match_visualization.png', dpi=150, bbox_inches='tight')
                print("Saved visualization to simple_match_visualization.png")
                plt.close()
        else:
            print("No keypoints in output")
            
    except Exception as e:
        print(f"Error during matching: {e}")
        import traceback
        traceback.print_exc()

# Also try using OpenCV's ORB as a fallback test
print("\n" + "="*50)
print("Testing with OpenCV ORB for comparison...")

orb = cv2.ORB_create(nfeatures=2000)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

print(f"ORB found {len(kp1)} keypoints in image 1")
print(f"ORB found {len(kp2)} keypoints in image 2")

if des1 is not None and des2 is not None:
    # Match with BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    print(f"ORB found {len(matches)} matches")
    
    if len(matches) > 0:
        # Draw matches
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, 
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite('orb_matches.png', img_matches)
        print("Saved ORB matches to orb_matches.png")