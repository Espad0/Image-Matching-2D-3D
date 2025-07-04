#!/usr/bin/env python3
"""
Debug script to inspect nested options within IncrementalPipelineOptions
"""

import pycolmap

def inspect_nested_mapper_options():
    """Inspect the mapper options within IncrementalPipelineOptions"""
    print("=== NESTED MAPPER OPTIONS ===\n")
    
    pipeline_options = pycolmap.IncrementalPipelineOptions()
    mapper_options = pipeline_options.mapper
    
    print("Type:", type(mapper_options))
    print("\nAttributes and values:")
    
    attrs = [attr for attr in dir(mapper_options) if not attr.startswith('_')]
    for attr in sorted(attrs):
        if hasattr(mapper_options, attr):
            try:
                value = getattr(mapper_options, attr)
                if not callable(value):
                    print(f"  {attr}: {value}")
            except:
                pass

def inspect_nested_triangulation_options():
    """Inspect the triangulation options within IncrementalPipelineOptions"""
    print("\n\n=== NESTED TRIANGULATION OPTIONS ===\n")
    
    pipeline_options = pycolmap.IncrementalPipelineOptions()
    triangulation_options = pipeline_options.triangulation
    
    print("Type:", type(triangulation_options))
    print("\nAttributes and values:")
    
    attrs = [attr for attr in dir(triangulation_options) if not attr.startswith('_')]
    for attr in sorted(attrs):
        if hasattr(triangulation_options, attr):
            try:
                value = getattr(triangulation_options, attr)
                if not callable(value):
                    print(f"  {attr}: {value}")
            except:
                pass

def test_setting_options():
    """Test setting various options"""
    print("\n\n=== TESTING OPTION SETTING ===\n")
    
    pipeline_options = pycolmap.IncrementalPipelineOptions()
    
    # Test setting pipeline-level options
    print("Before setting min_model_size:", pipeline_options.min_model_size)
    pipeline_options.min_model_size = 3
    print("After setting min_model_size:", pipeline_options.min_model_size)
    
    print("\nBefore setting ba_local_max_refinements:", pipeline_options.ba_local_max_refinements)
    pipeline_options.ba_local_max_refinements = 2
    print("After setting ba_local_max_refinements:", pipeline_options.ba_local_max_refinements)
    
    print("\nBefore setting ba_global_max_refinements:", pipeline_options.ba_global_max_refinements)
    pipeline_options.ba_global_max_refinements = 20
    print("After setting ba_global_max_refinements:", pipeline_options.ba_global_max_refinements)
    
    # Test setting nested mapper options
    print("\nNested mapper options:")
    print("Before setting mapper.init_min_num_inliers:", pipeline_options.mapper.init_min_num_inliers)
    pipeline_options.mapper.init_min_num_inliers = 50
    print("After setting mapper.init_min_num_inliers:", pipeline_options.mapper.init_min_num_inliers)

if __name__ == "__main__":
    inspect_nested_mapper_options()
    inspect_nested_triangulation_options()
    test_setting_options()