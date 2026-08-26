"""Behavioural parity against the conda-forge pyflann package."""

from __future__ import annotations

import numpy as np
import pytest

import mojo_pyflann as ours
from mojo_pyflann._lib import f64

upstream = pytest.importorskip("pyflann")


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(32)
    points = np.ascontiguousarray(rng.normal(size=(257, 11)).astype(np.float32))
    queries = np.ascontiguousarray(rng.normal(size=(19, 11)).astype(np.float32))
    return points, queries


def reference_linear(points, queries, k):
    ref = upstream.FLANN(random_seed=3)
    ref.build_index(points, algorithm="linear")
    return ref.nn_index(queries, k, checks=-1)


@pytest.mark.parametrize("k", [1, 5, 17])
def test_linear_index_matches_upstream(data, k):
    points, queries = data
    index = ours.FLANN(random_seed=3)
    params = index.build_index(points, algorithm="linear")
    got_idx, got_dist = index.nn_index(queries, k, checks=-1)
    want_idx, want_dist = reference_linear(points, queries, k)
    assert params["algorithm"] == "linear"
    assert np.array_equal(got_idx, want_idx)
    assert np.allclose(got_dist, want_dist, rtol=2e-6, atol=2e-6)


def test_kdtree_unlimited_checks_is_exact_and_limited_is_valid(data):
    points, queries = data
    index = ours.FLANN(random_seed=8)
    index.build_index(points, algorithm="kdtree", trees=4, leaf_max_size=8)
    exact_idx, exact_dist = index.nn_index(queries, 5, checks=-1)
    want_idx, want_dist = reference_linear(points, queries, 5)
    assert np.array_equal(exact_idx, want_idx)
    assert np.allclose(exact_dist, want_dist, rtol=2e-6, atol=2e-6)
    approx_idx, approx_dist = index.nn_index(queries, 5, checks=8)
    assert np.all((0 <= approx_idx) & (approx_idx < len(points)))
    calculated = ((points[approx_idx] - queries[:, None, :]) ** 2).sum(axis=2)
    assert np.allclose(approx_dist, calculated, rtol=2e-6, atol=2e-6)
    assert np.mean(approx_idx[:, 0] == want_idx[:, 0]) >= 0.5


def test_linear_simd_tail_and_parallel_threshold_match_upstream():
    rng = np.random.default_rng(31)
    points = np.ascontiguousarray(rng.normal(size=(12_000, 11)).astype(np.float32))
    queries = np.ascontiguousarray(rng.normal(size=(8, 11)).astype(np.float32))
    index = ours.FLANN()
    index.build_index(points, algorithm="linear")
    ref_idx, ref_dist = reference_linear(points, queries, 3)
    for count in (7, 8):
        got_idx, got_dist = index.nn_index(queries[:count], 3, checks=-1)
        assert np.array_equal(got_idx, ref_idx[:count])
        assert np.allclose(got_dist, ref_dist[:count], rtol=2e-6, atol=2e-6)


def test_contiguous_float64_input_is_zero_copy():
    points = np.ones((4, 3), dtype=np.float64)
    assert f64(points) is points
    index = ours.FLANN()
    index.build_index(points, algorithm="linear")
    assert index._data is points


def test_radius_and_one_shot_nn_match_upstream(data):
    points, queries = data
    radius = 7.25
    index = ours.FLANN(random_seed=5)
    index.build_index(points, algorithm="kdtree", trees=3)
    got_idx, got_dist = index.nn_radius(queries[0], radius, checks=-1, sorted=True)
    ref = upstream.FLANN(random_seed=5)
    ref.build_index(points, algorithm="linear")
    want_idx, want_dist = ref.nn_radius(queries[0], radius, checks=-1, sorted=True)
    assert np.array_equal(got_idx, want_idx)
    assert np.allclose(got_dist, want_dist, rtol=2e-6, atol=2e-6)
    one_idx, one_dist = index.nn(points, queries, 3, algorithm="linear", checks=-1)
    want_idx, want_dist = reference_linear(points, queries, 3)
    assert np.array_equal(one_idx, want_idx)
    assert np.allclose(one_dist, want_dist, rtol=2e-6, atol=2e-6)


def test_manhattan_and_index_lifecycle(tmp_path, data):
    points, queries = data
    ours.set_distance_type("manhattan")
    index = ours.FLANN()
    index.build_index(points, algorithm="linear")
    idx, dist = index.nn_index(queries, 1)
    expected = np.abs(points[None] - queries[:, None]).sum(axis=2)
    assert np.array_equal(idx, expected.argmin(axis=1))
    assert np.allclose(dist, expected.min(axis=1))
    ours.set_distance_type("euclidean")
    index = ours.FLANN(random_seed=9)
    index.build_index(points[:100], algorithm="kdtree", trees=2)
    path = tmp_path / "index.flann"
    index.save_index(path)
    loaded = ours.FLANN()
    loaded.load_index(path, points[:100])
    assert np.array_equal(loaded.nn_index(queries, 3, checks=-1)[0], index.nn_index(queries, 3, checks=-1)[0])
    index.add_points(points[100:110])
    assert index.used_memory() > index._data.nbytes
    assert len(index._data) == 110
    index.remove_point(0)
    assert len(index._data) == 109
    index.delete_index()
    with pytest.raises(ours.FLANNException):
        index.nn_index(queries)


@pytest.mark.parametrize("algorithm", ["kdtree_single", "autotuned"])
def test_kdtree_aliases_and_float_boundary(data, algorithm):
    points, queries = data
    index = ours.FLANN(random_seed=2)
    params = index.build_index(points, algorithm=algorithm, trees=2)
    got_idx, got_dist = index.nn_index(queries, 3, checks=-1)
    want_idx, want_dist = reference_linear(points, queries, 3)
    assert params["algorithm"] == "kdtree"
    assert np.array_equal(got_idx, want_idx)
    assert np.allclose(got_dist, want_dist, rtol=2e-6, atol=2e-6)
    with pytest.raises(ours.FLANNException, match="floating-point"):
        index.build_index(points.astype(np.int64), algorithm="linear")
