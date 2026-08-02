"""pyflann-compatible index facade with a linear and randomized KD-forest backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ._lib import addr, f64, i64, lib


class FLANNException(RuntimeError):
    pass


_DISTANCE = "euclidean"
_METRICS = {"euclidean": 0, "l2": 0, "manhattan": 1, "l1": 1}


def set_distance_type(distance_type: str, order: int = 0) -> None:
    """Select the distance used by subsequently created indices.

    pyflann's L2 result and radius are squared Euclidean distances; this port
    preserves that convention. ``order`` is accepted for pyflann parity.
    """
    del order
    global _DISTANCE
    name = str(distance_type).lower()
    if name not in _METRICS:
        raise FLANNException("covered distances are euclidean/l2 and manhattan/l1")
    _DISTANCE = name


def _matrix(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.floating):
        raise FLANNException(f"{name} must be a floating-point 2D array")
    if not len(array) or not array.shape[1]:
        raise FLANNException(f"{name} must not be empty")
    try:
        return f64(array)
    except TypeError as exc:
        raise FLANNException(f"{name} {exc}") from exc


class FLANN:
    """Approximate nearest neighbours using a randomized KD-forest.

    ``algorithm='linear'`` is exact. ``algorithm='kdtree'`` is a balanced
    randomized KD forest searched best-bin-first; increasing ``checks`` visits
    more leaves, while ``checks=-1`` searches every leaf exactly.
    """

    def __init__(self, **kwargs: Any):
        self._data: np.ndarray | None = None
        self._algorithm = "kdtree"
        self._metric_name = str(kwargs.pop("distance_type", _DISTANCE)).lower()
        if self._metric_name not in _METRICS:
            raise FLANNException("covered distances are euclidean/l2 and manhattan/l1")
        self._params: dict[str, Any] = {
            "algorithm": "kdtree", "checks": 32, "eps": 0.0, "sorted": 1,
            "max_neighbors": -1, "trees": 4, "leaf_max_size": 16,
            "random_seed": kwargs.pop("random_seed", None),
        }
        self._params.update(kwargs)
        self._tree: dict[str, np.ndarray] | None = None

    def _build_forest(self, trees: int, leaf_size: int, seed: int | None) -> dict[str, np.ndarray]:
        assert self._data is not None
        data = self._data
        rng = np.random.default_rng(seed)
        dims: list[int] = []
        splits: list[float] = []
        left: list[int] = []
        right: list[int] = []
        starts: list[int] = []
        ends: list[int] = []
        permutation: list[int] = []
        roots: list[int] = []

        def node(ids: np.ndarray) -> int:
            here = len(dims)
            dims.append(-1)
            splits.append(0.0)
            left.append(-1)
            right.append(-1)
            starts.append(-1)
            ends.append(-1)
            if len(ids) <= leaf_size:
                starts[here] = len(permutation)
                permutation.extend(ids.tolist())
                ends[here] = len(permutation)
                return here
            variance = np.var(data[ids], axis=0)
            choices = np.argsort(variance)[-min(5, data.shape[1]):]
            axis = int(choices[rng.integers(len(choices))])
            middle = len(ids) // 2
            arranged = ids[np.argpartition(data[ids, axis], middle)]
            dims[here] = axis
            splits[here] = float(data[arranged[middle], axis])
            left[here] = node(arranged[:middle])
            right[here] = node(arranged[middle:])
            return here

        initial = np.arange(len(data), dtype=np.int64)
        for _ in range(trees):
            roots.append(node(initial))
        return {
            "dim": i64(dims), "split": f64(splits), "left": i64(left),
            "right": i64(right), "start": i64(starts), "end": i64(ends),
            "perm": i64(permutation), "roots": i64(roots),
        }

    def build_index(self, pts, **kwargs: Any) -> dict[str, Any]:
        data = _matrix(pts, "pts")
        params = dict(self._params)
        params.update(kwargs)
        algorithm = str(params.get("algorithm", "kdtree")).lower()
        algorithm = {"kdtree_single": "kdtree", "autotuned": "kdtree"}.get(algorithm, algorithm)
        if algorithm not in {"linear", "kdtree"}:
            raise FLANNException("covered algorithms are linear, kdtree, kdtree_single, and autotuned")
        trees = max(1, int(params.get("trees", 4)))
        leaf_size = max(1, int(params.get("leaf_max_size", 16)))
        self._data, self._algorithm = data, algorithm
        self._params = params | {"algorithm": algorithm, "trees": trees, "leaf_max_size": leaf_size}
        self._tree = None if algorithm == "linear" else self._build_forest(trees, leaf_size, params.get("random_seed"))
        return dict(self._params) | {"speedup": 1.0 if algorithm == "kdtree" else 0.0}

    def _require_index(self) -> np.ndarray:
        if self._data is None:
            raise FLANNException("build_index(...) method not called first or current index deleted.")
        return self._data

    def _query(self, qpts, num_neighbors: int, **kwargs: Any):
        data = self._require_index()
        query = _matrix(qpts, "qpts")
        if query.shape[1] != data.shape[1]:
            raise FLANNException("data and query must have the same dims")
        k = int(num_neighbors)
        if not 1 <= k <= len(data):
            raise FLANNException("more neighbors than there are points")
        idx = np.empty((len(query), k), dtype=np.int64)
        dist = np.empty((len(query), k), dtype=np.float64)
        metric = _METRICS[self._metric_name]
        checks = int(kwargs.get("checks", self._params.get("checks", 32)))
        if self._algorithm == "linear" or checks < 0:
            lib().mpf_knn_linear(addr(data), addr(query), addr(idx), addr(dist), len(data), data.shape[1], len(query), k, metric)
        else:
            assert self._tree is not None
            tree = self._tree
            seen = np.empty(len(data), dtype=np.uint8)
            stack_node = np.empty(len(tree["dim"]), dtype=np.int64)
            stack_bound = np.empty(len(tree["dim"]), dtype=np.float64)
            lib().mpf_knn_kdtree(
                addr(data), addr(query), addr(tree["dim"]), addr(tree["split"]),
                addr(tree["left"]), addr(tree["right"]), addr(tree["start"]), addr(tree["end"]),
                addr(tree["perm"]), addr(tree["roots"]), addr(idx), addr(dist), addr(seen),
                addr(stack_node), addr(stack_bound), len(data), data.shape[1], len(query), k,
                len(tree["roots"]), checks, metric,
            )
        if k == 1:
            return idx[:, 0], dist[:, 0]
        return idx, dist

    def nn_index(self, qpts, num_neighbors: int = 1, **kwargs: Any):
        return self._query(qpts, num_neighbors, **kwargs)

    def nn(self, pts, qpts, num_neighbors: int = 1, **kwargs: Any):
        index = FLANN(distance_type=self._metric_name, **self._params)
        index.build_index(pts, **kwargs)
        return index.nn_index(qpts, num_neighbors, **kwargs)

    def nn_radius(self, query, radius: float, **kwargs: Any):
        data = self._require_index()
        point = _matrix(query, "query")
        if len(point) != 1 or point.shape[1] != data.shape[1]:
            raise FLANNException("data and query must have the same dims")
        idx = np.empty(len(data), dtype=np.int64)
        dist = np.empty(len(data), dtype=np.float64)
        count = lib().mpf_radius_linear(
            addr(data), addr(point), addr(idx), addr(dist), len(data), data.shape[1],
            float(radius), _METRICS[self._metric_name], int(kwargs.get("sorted", self._params.get("sorted", 1))), -1,
        )
        limit = int(kwargs.get("max_neighbors", self._params.get("max_neighbors", -1)))
        if limit >= 0:
            count = min(count, limit)
        return idx[:count], dist[:count]

    def delete_index(self, **kwargs: Any) -> None:
        self._params.update(kwargs)
        self._data = self._tree = None

    def used_memory(self) -> int:
        data = self._require_index()
        return int(data.nbytes + sum(v.nbytes for v in (self._tree or {}).values()))

    def add_points(self, pts, rebuild_threshold: float = 2.0):
        del rebuild_threshold
        data = self._require_index()
        extra = _matrix(pts, "pts")
        if extra.shape[1] != data.shape[1]:
            raise FLANNException("data and query must have the same dims")
        self.build_index(np.vstack((data, extra)), **self._params)

    def remove_point(self, idx: int):
        data = self._require_index()
        self.build_index(np.delete(data, int(idx), axis=0), **self._params)

    def save_index(self, filename):
        self._require_index()
        with Path(filename).open("wb") as handle:
            np.savez(handle, algorithm=self._algorithm, params=np.array([self._params], dtype=object))

    def load_index(self, filename, pts):
        with np.load(filename, allow_pickle=True) as saved:
            params = dict(saved["params"][0])
            params["algorithm"] = str(saved["algorithm"])
        return self.build_index(pts, **params)
