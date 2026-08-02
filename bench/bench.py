"""Benchmark mojo-pyflann against pyflann on identical float32 arrays."""

from __future__ import annotations

import math
import platform
import time

import numpy as np
import pyflann

from mojo_pyflann import FLANN


def best(fn, repeat: int = 3) -> float:
    value = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        value = min(value, time.perf_counter() - start)
    return value


def arrays(n: int, q: int, d: int):
    rng = np.random.default_rng(10)
    return (
        np.ascontiguousarray(rng.normal(size=(n, d)).astype(np.float32)),
        np.ascontiguousarray(rng.normal(size=(q, d)).astype(np.float32)),
    )


def measure(name: str, data: np.ndarray, query: np.ndarray, algorithm: str, checks: int, ref_algorithm: str | None = None):
    ours = FLANN(random_seed=4)
    ours.build_index(data, algorithm=algorithm, trees=4, leaf_max_size=16)
    ref = pyflann.FLANN(random_seed=4)
    ref.build_index(data, algorithm=ref_algorithm or algorithm, trees=1 if (ref_algorithm or algorithm) == "linear" else 4, leaf_max_size=16)
    ours.nn_index(query[:1], 5, checks=checks)
    ref.nn_index(query[:1], 5, checks=checks)
    a = best(lambda: ours.nn_index(query, 5, checks=checks))
    b = best(lambda: ref.nn_index(query, 5, checks=checks))
    print(f"| {name} | {a * 1e3:.1f} ms | {b * 1e3:.1f} ms | {b / a:.2f}x {'faster' if a < b else 'slower'} |")


def main():
    print(f"Machine: {platform.platform()} ({platform.processor() or 'processor unavailable'})")
    print("| case | mojo-pyflann | pyflann | ratio |")
    print("| --- | ---: | ---: | --- |")
    data, query = arrays(8_000, 400, 24)
    measure("linear exact query: 8k x 400, 24d, k=5", data, query, "linear", -1)
    data, query = arrays(30_000, 1_000, 32)
    measure("KD forest query: 30k x 1k, 32d, k=5, checks=32", data, query, "kdtree", 32)
    exact_data, exact_query = arrays(8_000, 200, 32)
    measure("KD forest exact query: 8k x 200, 32d, k=5", exact_data, exact_query, "kdtree", -1, "linear")


if __name__ == "__main__":
    main()
