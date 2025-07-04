#!/usr/bin/env python3
"""
Debug script to inspect pycolmap IncrementalMapperOptions attributes
"""

import pycolmap

def inspect_mapper_options():
    """Inspect and print all attributes of IncrementalMapperOptions"""
    print("Inspecting pycolmap.IncrementalMapperOptions\n")
    
    # Create an instance of IncrementalMapperOptions
    mapper_options = pycolmap.IncrementalMapperOptions()
    
    # Get all attributes
    print("=== All attributes (dir()) ===")
    all_attrs = dir(mapper_options)
    for attr in sorted(all_attrs):
        if not attr.startswith('_'):  # Skip private/magic methods
            print(f"  {attr}")
    
    print("\n=== Attribute values ===")
    # Try to get values for each attribute
    for attr in sorted(all_attrs):
        if not attr.startswith('_'):
            try:
                value = getattr(mapper_options, attr)
                # Skip methods
                if not callable(value):
                    print(f"  {attr}: {value} (type: {type(value).__name__})")
            except Exception as e:
                print(f"  {attr}: <Error accessing: {e}>")
    
    print("\n=== Check for specific attributes ===")
    # Check for specific attributes we're interested in
    specific_attrs = [
        'min_model_size',
        'min_num_matches',
        'ignore_watermarks',
        'multiple_models',
        'max_extra_param',
        'ba_refine_focal_length',
        'ba_refine_principal_point',
        'ba_refine_extra_params',
        'ba_local_max_refinements',
        'ba_global_max_refinements',
        'init_min_num_inliers',
        'abs_pose_min_num_inliers',
        'abs_pose_min_inlier_ratio',
        'filter_max_reproj_error',
        'filter_min_tri_angle',
        'max_reg_trials'
    ]
    
    for attr in specific_attrs:
        if hasattr(mapper_options, attr):
            try:
                value = getattr(mapper_options, attr)
                print(f"  ✓ {attr}: {value}")
            except Exception as e:
                print(f"  ✗ {attr}: <Error: {e}>")
        else:
            print(f"  ✗ {attr}: NOT FOUND")
    
    print("\n=== Type information ===")
    print(f"Type: {type(mapper_options)}")
    print(f"Module: {type(mapper_options).__module__}")
    
    # Try to access help/docstring
    print("\n=== Documentation ===")
    try:
        help_text = mapper_options.__doc__
        if help_text:
            print(f"Docstring: {help_text}")
        else:
            print("No docstring available")
    except:
        print("Could not access documentation")

def inspect_bundle_adjustment_options():
    """Inspect and print all attributes of BundleAdjustmentOptions"""
    print("\n\n=== BUNDLE ADJUSTMENT OPTIONS ===")
    print("Inspecting pycolmap.BundleAdjustmentOptions\n")
    
    try:
        # Create an instance of BundleAdjustmentOptions
        ba_options = pycolmap.BundleAdjustmentOptions()
        
        # Get all attributes
        print("=== All attributes (dir()) ===")
        all_attrs = dir(ba_options)
        for attr in sorted(all_attrs):
            if not attr.startswith('_'):  # Skip private/magic methods
                print(f"  {attr}")
        
        print("\n=== Attribute values ===")
        # Try to get values for each attribute
        for attr in sorted(all_attrs):
            if not attr.startswith('_'):
                try:
                    value = getattr(ba_options, attr)
                    # Skip methods
                    if not callable(value):
                        print(f"  {attr}: {value} (type: {type(value).__name__})")
                except Exception as e:
                    print(f"  {attr}: <Error accessing: {e}>")
    except Exception as e:
        print(f"Error creating BundleAdjustmentOptions: {e}")

def inspect_reconstruction_options():
    """Check if there's a ReconstructionOptions or similar"""
    print("\n\n=== CHECKING OTHER OPTION CLASSES ===")
    
    # List of potential option classes to check
    option_classes = [
        'ReconstructionOptions',
        'MapperOptions',
        'IncrementalMapperOptions',
        'BundleAdjustmentOptions',
        'TriangulationOptions',
        'IncrementalPipelineOptions',
        'PipelineOptions'
    ]
    
    for class_name in option_classes:
        if hasattr(pycolmap, class_name):
            print(f"✓ Found: pycolmap.{class_name}")
        else:
            print(f"✗ Not found: pycolmap.{class_name}")

def inspect_incremental_pipeline_options():
    """Inspect IncrementalPipelineOptions"""
    print("\n\n=== INCREMENTAL PIPELINE OPTIONS ===")
    print("Inspecting pycolmap.IncrementalPipelineOptions\n")
    
    try:
        # Create an instance of IncrementalPipelineOptions
        pipeline_options = pycolmap.IncrementalPipelineOptions()
        
        # Get all attributes
        print("=== All attributes (dir()) ===")
        all_attrs = dir(pipeline_options)
        for attr in sorted(all_attrs):
            if not attr.startswith('_'):  # Skip private/magic methods
                print(f"  {attr}")
        
        print("\n=== Attribute values ===")
        # Try to get values for each attribute
        for attr in sorted(all_attrs):
            if not attr.startswith('_'):
                try:
                    value = getattr(pipeline_options, attr)
                    # Skip methods
                    if not callable(value):
                        # Special handling for nested objects
                        if hasattr(value, '__dict__') or hasattr(value, 'todict'):
                            print(f"  {attr}: <{type(value).__name__} object>")
                        else:
                            print(f"  {attr}: {value} (type: {type(value).__name__})")
                except Exception as e:
                    print(f"  {attr}: <Error accessing: {e}>")
        
        # Check for specific nested options
        print("\n=== Checking nested options ===")
        if hasattr(pipeline_options, 'mapper'):
            print("  Found 'mapper' attribute - this might contain mapper options")
        if hasattr(pipeline_options, 'bundle_adjustment'):
            print("  Found 'bundle_adjustment' attribute - this might contain BA options")
            
    except Exception as e:
        print(f"Error creating IncrementalPipelineOptions: {e}")

if __name__ == "__main__":
    inspect_mapper_options()
    inspect_bundle_adjustment_options()
    inspect_reconstruction_options()
    inspect_incremental_pipeline_options()