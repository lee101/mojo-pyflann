"""ctypes loading for the Mojo nearest-neighbour kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-pyflann.so")
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mpf_knn_linear": ([I] * 9, None),
    "mpf_knn_kdtree": ([I] * 22, None),
    "mpf_radius_linear": ([I, I, I, I, I, I, F, I, I, I], I),
}


def build() -> str:
    """Build the shared object when it is absent or older than the kernel."""
    source = os.path.join(ROOT, "src", "capi.mojo")
    if os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    compiler = shutil.which("mojo")
    if not compiler:
        raise RuntimeError("Mojo is unavailable; run through `pixi run ...`")
    subprocess.run([os.path.join(ROOT, "build", "build.sh")], check=True, timeout=1800)
    return LIB


_loaded: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_loaded, name)
            fn.argtypes, fn.restype = argtypes, restype
    return _loaded


def f64(a) -> np.ndarray:
    array = np.asarray(a)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("nearest-neighbour inputs must have a floating-point dtype")
    if not np.can_cast(array.dtype, np.float64, casting="safe"):
        raise TypeError(f"{array.dtype} cannot be represented safely as float64")
    return np.ascontiguousarray(array, dtype=np.float64)


def i64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int64)


def addr(a: np.ndarray) -> int:
    if not isinstance(a, np.ndarray) or not a.flags.c_contiguous or not a.size:
        raise ValueError("FFI buffers must be non-empty C-contiguous numpy arrays")
    address = int(a.ctypes.data)
    if not address:
        raise ValueError("FFI buffers must have a non-null data pointer")
    return address
