# mojo-pyflann

`mojo-pyflann` is a standalone Mojo port of the compute-bound nearest-neighbour core of [pyflann](https://github.com/primetang/pyflann). Its Python package, `mojo_pyflann`, mirrors the covered `pyflann.FLANN` workflow: build an index, query it with `nn_index` or `nn_radius`, and manage it with save/load and point updates.

## Coverage

| API / algorithm | status |
| --- | --- |
| `FLANN.build_index`, `nn_index`, `nn`, `nn_radius` | implemented |
| `linear` | exact k-nearest-neighbour search in Mojo |
| `kdtree` | randomized balanced KD-forest with best-bin-first approximate search |
| `kdtree_single`, `autotuned` | accepted as compatibility aliases for `kdtree`; no single-tree or auto-tuning policy |
| `checks=-1` | exact KD-forest search |
| `euclidean` / `l2`, `manhattan` / `l1` | implemented; L2 results and radii are squared, as in pyflann |
| `save_index`, `load_index`, `add_points`, `remove_point`, `delete_index`, `used_memory` | implemented |

Not covered: hierarchical k-means, composite, LSH and the other FLANN index types; `kmeans`/`hierarchical_kmeans`; the distinct `kdtree_single` and auto-tuning behaviours; non-L1/L2 metrics; and binary-compatible loading of a native FLANN index file. `load_index` restores this port's compact metadata and rebuilds its KD forest around the supplied data, which preserves the upstream method signature without pretending its binary index format is portable.

## Install and use

```bash
pixi install
pixi run build
pixi run test
```

`pixi` activates `python/` automatically for its tasks. This example runs as-is with `pixi run python example.py`:

```python
import numpy as np
from mojo_pyflann import FLANN

points = np.ascontiguousarray([[0., 0.], [1., 0.], [0., 2.], [4., 4.]])
queries = np.ascontiguousarray([[.2, .1], [3., 3.]])

index = FLANN(random_seed=7)
index.build_index(points, algorithm="kdtree", trees=4, leaf_max_size=8)
indices, squared_distances = index.nn_index(queries, num_neighbors=2, checks=32)
print(indices)
# [[0 1]
#  [3 2]]
```

Choose `algorithm="linear"` for an exact scan. For KD forests, `checks` is the maximum number of leaves examined: higher values improve recall; `checks=-1` examines every leaf and is exact. This is intentionally a direct control, rather than an invented promise about recall at a fixed setting.

## Benchmark

Measured by `pixi run bench` on this machine (Linux 6.8.0-136-generic, x86_64) using identical float32 arrays. Each value is the best of three query-only runs; index construction is excluded.

| case | mojo-pyflann | pyflann | ratio |
| --- | ---: | ---: | --- |
| linear exact query: 8k x 400, 24d, k=5 | 4.9 ms | 7.1 ms | 1.46x faster |
| KD forest query: 30k x 1k, 32d, k=5, checks=32 | 6.1 ms | 8.8 ms | 1.45x faster |
| KD forest exact query: 8k x 200, 32d, k=5 | 7.9 ms | 6.2 ms | 0.78x slower |

The L1 and L2 distance kernels use host-width float64 SIMD with scalar remainders. Large independent query batches are split across a bounded 16-worker pool; smaller calls remain serial to avoid launch overhead. KD best-bin-first candidates use a binary min-heap instead of rescanning the candidate stack. An unlimited KD query is an exact scan, so it uses the linear kernel directly instead of traversing every KD leaf. The exact case remains slower than native FLANN in this run and is reported as such.

There is no GPU path. L2 distance performs roughly three floating-point operations per 16 bytes of input loaded (about 0.19 flop/byte), L1 is lower, and KD traversal adds irregular memory traffic. None of these kernels approaches the requested 2 flop/byte threshold, so transfer and launch costs do not justify GPU execution.

## How it works

The Python facade preserves C-contiguous `float64` NumPy buffers zero-copy and converts other floating-point inputs once to the kernel's `float64` representation. It owns the flat KD-forest arrays. Integer and precision-losing input dtypes are rejected before the ABI call. A tree node stores its split dimension/value, two child indices, or a range into an `int64` point permutation. Query buffers cross `ctypes` as borrowed addresses; the C ABI rebuilds typed pointers inside Mojo and neither allocates nor owns Python memory.

The linear kernel scans the row-major `(n_samples, n_features)` array and keeps a sorted `k`-element result list. The KD kernel uses a caller-provided best-bin-first heap, marks already inspected points, and searches leaf ranges. All results are sorted nearest first.

## Development

```bash
pixi run build && pixi run test && pixi run bench
```

The tests assert numerical and behavioural parity with installed upstream `pyflann` for exact linear, exact KD, radius, one-shot queries, aliases, and index lifecycle operations. They also verify approximate candidates are valid distances and reject integer input at the FFI boundary.

## License

MIT
