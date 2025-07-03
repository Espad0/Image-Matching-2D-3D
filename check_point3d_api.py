"""Check pycolmap Point3D API."""
import pycolmap

print("Available in pycolmap._core.Point3D:")
print([attr for attr in dir(pycolmap._core.Point3D) if not attr.startswith('_')])