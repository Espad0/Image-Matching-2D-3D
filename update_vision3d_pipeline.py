#!/usr/bin/env python3
"""
Script to update Vision3D pipeline with improvements from IMC notebook.
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_loftr_matcher():
    """Update LoFTR matcher with multi-resolution and TTA support."""
    
    loftr_path = Path("vision3d/models/loftr.py")
    
    # Add the match_pair_multires method
    multires_method = '''
    def match_pair_multires(
        self,
        image1_path: str,
        image2_path: str,
        resolutions: List[int] = None,
        use_tta: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Enhanced matching with multiple resolutions and TTA.
        Based on IMC-2023 winning solution.
        """
        if resolutions is None:
            resolutions = [1024, 1440]  # IMC notebook resolutions
            
        all_mkpts1 = []
        all_mkpts2 = []
        all_mconf = []
        
        for resolution in resolutions:
            # Standard forward matching
            mkpts1, mkpts2, mconf = self.match_pair(
                image1_path, image2_path, resize=resolution
            )
            all_mkpts1.append(mkpts1)
            all_mkpts2.append(mkpts2)
            all_mconf.append(mconf)
            
            logger.debug(f"Resolution {resolution}: {len(mkpts1)} matches")
            
            # Test-time augmentation with horizontal flip
            if use_tta and len(mkpts1) > 0:
                mkpts1_flip, mkpts2_flip, mconf_flip = self._match_with_flip(
                    image1_path, image2_path, resolution
                )
                if len(mkpts1_flip) > 0:
                    all_mkpts1.append(mkpts1_flip)
                    all_mkpts2.append(mkpts2_flip)
                    all_mconf.append(mconf_flip)
                    logger.debug(f"TTA flip: {len(mkpts1_flip)} matches")
        
        # Bidirectional matching (reverse order)
        mkpts2_rev, mkpts1_rev, mconf_rev = self.match_pair(
            image2_path, image1_path, resize=resolutions[0]
        )
        if len(mkpts1_rev) > 0:
            all_mkpts1.append(mkpts1_rev)
            all_mkpts2.append(mkpts2_rev)
            all_mconf.append(mconf_rev)
            logger.debug(f"Bidirectional: {len(mkpts1_rev)} matches")
        
        # Concatenate all matches
        if not all_mkpts1:
            return np.array([]), np.array([]), np.array([])
            
        mkpts1 = np.concatenate(all_mkpts1, axis=0)
        mkpts2 = np.concatenate(all_mkpts2, axis=0)
        mconf = np.concatenate(all_mconf, axis=0)
        
        logger.info(f"Total matches before deduplication: {len(mkpts1)}")
        
        # Remove duplicates
        unique_matches = self._deduplicate_matches(mkpts1, mkpts2, mconf)
        mkpts1, mkpts2, mconf = unique_matches
        
        logger.info(f"Total matches after deduplication: {len(mkpts1)}")
        
        return mkpts1, mkpts2, mconf
    
    def _match_with_flip(
        self,
        image1_path: str,
        image2_path: str,
        resolution: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Match with horizontal flip augmentation."""
        # Load and flip images
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Temporary flipped tensors
        img1_tensor = self._preprocess_image(image1_path, resolution)[1]
        img2_tensor = self._preprocess_image(image2_path, resolution)[1]
        
        # Flip tensors
        img1_flip = torch.flip(img1_tensor, dims=[3])
        img2_flip = torch.flip(img2_tensor, dims=[3])
        
        # Match flipped images
        with torch.no_grad():
            mkpts1, mkpts2, mconf = self._match_single(img1_flip, img2_flip)
        
        # Adjust keypoint coordinates back
        scale1 = resolution / max(h1, w1) if resolution else 1.0
        scale2 = resolution / max(h2, w2) if resolution else 1.0
        
        # Unflip x-coordinates
        mkpts1[:, 0] = (resolution if resolution else w1) - mkpts1[:, 0] - 1
        mkpts2[:, 0] = (resolution if resolution else w2) - mkpts2[:, 0] - 1
        
        # Rescale to original size
        mkpts1 = mkpts1 / scale1
        mkpts2 = mkpts2 / scale2
        
        return mkpts1, mkpts2, mconf
    
    def _deduplicate_matches(
        self,
        mkpts1: np.ndarray,
        mkpts2: np.ndarray,
        mconf: np.ndarray,
        pixel_threshold: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Remove duplicate matches within pixel threshold."""
        if len(mkpts1) == 0:
            return mkpts1, mkpts2, mconf
        
        # Round keypoints for deduplication
        mkpts_rounded = np.round(np.concatenate([mkpts1, mkpts2], axis=1))
        
        # Find unique matches
        _, unique_idx = np.unique(mkpts_rounded, axis=0, return_index=True)
        
        # Sort by confidence and keep best matches
        unique_idx = unique_idx[np.argsort(mconf[unique_idx])[::-1]]
        
        return mkpts1[unique_idx], mkpts2[unique_idx], mconf[unique_idx]
'''
    
    logger.info(f"Add the following methods to {loftr_path}:")
    print(multires_method)
    
    # Update the default config
    config_update = '''
    # Update the default configuration in LoFTRMatcher.__init__
    self.config = {
        'pretrained': 'outdoor',
        'confidence_threshold': 0.3,  # IMC uses 0.3
        'image_resize': 1024,  # IMC uses 1024/1440
        'use_tta': True,  # Enable test-time augmentation
    }
'''
    print("\nConfiguration updates:")
    print(config_update)


def update_pipeline_config():
    """Update pipeline configuration based on IMC notebook."""
    
    pipeline_updates = '''
# In vision3d/core/pipeline.py, update the _extract_and_match method:

def _extract_and_match(
    self,
    image_paths: List[str],
    image_pairs: List[Tuple[int, int]],
    feature_dir: Path,
    verbose: bool = True
) -> Dict:
    """Extract features and perform matching with IMC improvements."""
    feature_dir.mkdir(parents=True, exist_ok=True)
    
    all_matches = {}
    
    # Use IMC strategy based on number of pairs
    if len(image_pairs) >= 400:
        # Large scenes: SuperGlue only (if available)
        logger.info(f"Large scene ({len(image_pairs)} pairs) - using SuperGlue")
        if hasattr(self, 'superglue_matcher'):
            matches = self.superglue_matcher.match_pairs(
                image_paths,
                image_pairs,
                feature_dir,
                verbose=verbose,
                resolutions=[1024, 1440]  # Multi-resolution
            )
            all_matches.update(matches)
        else:
            # Fall back to LoFTR
            logger.info("SuperGlue not available, using LoFTR")
            matches = self.loftr_matcher.match_pairs_multires(
                image_paths,
                image_pairs,
                feature_dir,
                verbose=verbose
            )
            all_matches.update(matches)
    else:
        # Small scenes: Combine LoFTR + SuperGlue
        logger.info(f"Small scene ({len(image_pairs)} pairs) - combining matchers")
        
        # LoFTR with multi-resolution and TTA
        matches = self.loftr_matcher.match_pairs_multires(
            image_paths,
            image_pairs,
            feature_dir,
            verbose=verbose,
            resolutions=[1024],  # Single resolution for speed
            use_tta=True
        )
        all_matches.update(matches)
        
        # Add SuperGlue if available
        if hasattr(self, 'superglue_matcher'):
            sg_matches = self.superglue_matcher.match_pairs(
                image_paths,
                image_pairs,
                feature_dir,
                verbose=verbose,
                resolutions=[1024, 1440]
            )
            # Merge matches
            for pair, match_data in sg_matches.items():
                if pair in all_matches:
                    # Concatenate matches
                    all_matches[pair]['keypoints1'] = np.vstack([
                        all_matches[pair]['keypoints1'],
                        match_data['keypoints1']
                    ])
                    all_matches[pair]['keypoints2'] = np.vstack([
                        all_matches[pair]['keypoints2'],
                        match_data['keypoints2']
                    ])
                else:
                    all_matches[pair] = match_data
    
    return all_matches
'''
    
    print("\nPipeline updates:")
    print(pipeline_updates)
    
    # Reconstruction parameters
    recon_params = '''
# In vision3d/core/pipeline.py, update the reconstruct method:

# Add custom mapper options based on IMC notebook
mapper_options = pycolmap.IncrementalMapperOptions()
mapper_options.min_model_size = 3
mapper_options.ba_local_max_refinements = 2
mapper_options.ba_global_max_refinements = 20
mapper_options.ba_global_max_num_iterations = 50
mapper_options.init_max_error = 4.0
mapper_options.filter_max_reproj_error = 4.0
mapper_options.filter_min_tri_angle = 1.5

# Pass options to reconstruction engine
reconstruction = self.reconstruction_engine.reconstruct(
    database_path,
    Path(image_paths[0]).parent,
    output_dir / 'sparse',
    verbose=verbose,
    options=mapper_options  # Add this parameter
)
'''
    
    print("\nReconstruction parameters:")
    print(recon_params)


def create_updated_loftr_module():
    """Create a complete updated LoFTR module file."""
    
    updated_file = Path("vision3d/models/loftr_improved.py")
    logger.info(f"Creating improved LoFTR module: {updated_file}")
    
    # This would contain the full updated module
    # For now, just indicate what needs to be done
    print(f"\nTo apply improvements:")
    print(f"1. Add match_pair_multires method to LoFTRMatcher class")
    print(f"2. Add _match_with_flip method for TTA")  
    print(f"3. Add _deduplicate_matches method")
    print(f"4. Update default config with higher resolution and confidence threshold")
    print(f"5. Update pipeline to use multi-resolution matching")
    print(f"6. Add custom COLMAP mapper options")


if __name__ == "__main__":
    print("=== Vision3D Pipeline Improvements from IMC-2023 ===\n")
    
    print("Key improvements to implement:")
    print("1. Multi-resolution matching (1024, 1440 pixels)")
    print("2. Test-time augmentation (horizontal flip)")
    print("3. Bidirectional matching")
    print("4. Better match deduplication")
    print("5. Custom COLMAP parameters")
    print("6. Adaptive matcher selection based on scene size\n")
    
    update_loftr_matcher()
    update_pipeline_config()
    create_updated_loftr_module()
    
    print("\n=== Summary ===")
    print("The IMC notebook achieves better results through:")
    print("- Higher resolution processing (1024/1440 vs 640 pixels)")
    print("- Multiple matching strategies combined")
    print("- Test-time augmentation for more matches")
    print("- Optimized COLMAP parameters for speed/quality")
    print("- Intelligent scene-based matcher selection")