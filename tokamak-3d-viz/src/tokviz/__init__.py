"""tokviz — physics bake pipeline for the tokamak 3D visualization.

Runs in a normal CPython venv (numpy/scipy/pyarrow/skimage).  Nothing in this
package imports ``bpy``; Blender only ever consumes the ``.npy``/``.json``
artifacts written by :mod:`tokviz.export`.
"""
__version__ = "0.1.0"
