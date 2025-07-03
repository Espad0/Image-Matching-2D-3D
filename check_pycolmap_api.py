"""Check pycolmap API to understand the correct method names."""
import pycolmap

# Create a dummy image to check available methods
print("pycolmap version:", pycolmap.__version__)

# Check what attributes are available
print("\nChecking pycolmap Image attributes:")
# We can't create an Image directly, so let's check the documentation
print("Available in pycolmap._core.Image:")
print([attr for attr in dir(pycolmap._core.Image) if not attr.startswith('_')])