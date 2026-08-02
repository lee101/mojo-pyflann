"""A compact, Mojo-accelerated subset of pyflann."""

from .index import FLANN, FLANNException, set_distance_type

__version__ = "0.1.0"
__all__ = ["FLANN", "FLANNException", "set_distance_type"]
