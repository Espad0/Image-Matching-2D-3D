"""Check available pycolmap functions."""

import pycolmap

print("Available pycolmap attributes:")
attrs = [attr for attr in dir(pycolmap) if not attr.startswith('_')]
for attr in sorted(attrs):
    print(f"  {attr}")

# Check for mapping functions
print("\nMapping/reconstruction related:")
mapping_attrs = [attr for attr in attrs if 'map' in attr.lower() or 'reconstruct' in attr.lower()]
for attr in sorted(mapping_attrs):
    print(f"  {attr}")

# Check incremental mapper
print("\nTrying to use incremental_mapping...")
try:
    from pycolmap import incremental_mapping
    print("incremental_mapping is available")
except ImportError:
    print("incremental_mapping not found")

# Check what's in the module
print("\nChecking _core module:")
if hasattr(pycolmap, '_core'):
    core_attrs = [attr for attr in dir(pycolmap._core) if not attr.startswith('_')]
    for attr in sorted(core_attrs)[:20]:  # First 20
        print(f"  {attr}")