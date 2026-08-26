"""Hot nearest-neighbour loops exported to Python through a tiny C ABI."""

from std.sys.info import simd_width_of
from std.algorithm import map

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime BPtr = Pointer[UInt8, AnyOrigin[mut=True]]


def distance(a: Ptr, b: Ptr, d: Int, metric: Int) -> Float64:
    var total = 0.0
    if metric == 1:
        for j in range(d):
            var delta = a[unsafe_offset=j] - b[unsafe_offset=j]
            total += -delta if delta < 0.0 else delta
    else:
        comptime W = simd_width_of[DType.float64]()
        var j = 0
        while j + W <= d:
            var delta = a.unsafe_load[width=W](j) - b.unsafe_load[width=W](j)
            total += (delta * delta).reduce_add()
            j += W
        for j in range(j, d):
            var delta = a[unsafe_offset=j] - b[unsafe_offset=j]
            total += delta * delta
    return total


def insert_neighbour(idx: IPtr, dist: Ptr, base: Int, k: Int, point: Int, value: Float64):
    if value >= dist[unsafe_offset=base + k - 1]:
        return
    var slot = k - 1
    while slot > 0 and dist[unsafe_offset=base + slot - 1] > value:
        dist[unsafe_offset=base + slot] = dist[unsafe_offset=base + slot - 1]
        idx[unsafe_offset=base + slot] = idx[unsafe_offset=base + slot - 1]
        slot -= 1
    dist[unsafe_offset=base + slot] = value
    idx[unsafe_offset=base + slot] = Int64(point)


def linear_query(
    train: Ptr, query: Ptr, idx: IPtr, dist: Ptr,
    n: Int, d: Int, k: Int, metric: Int, q: Int,
):
    var base = q * k
    for s in range(k):
        idx[unsafe_offset=base + s] = -1
        dist[unsafe_offset=base + s] = 1.7976931348623157e308
    for row in range(n):
        insert_neighbour(idx, dist, base, k, row, distance(query.unsafe_offset(q * d), train.unsafe_offset(row * d), d, metric))


@export("mpf_knn_linear")
def mpf_knn_linear(
    train_addr: Int, query_addr: Int, idx_addr: Int, dist_addr: Int,
    n: Int, d: Int, m: Int, k: Int, metric: Int,
) abi("C"):
    var train = Ptr(unsafe_from_address=train_addr)
    var query = Ptr(unsafe_from_address=query_addr)
    var idx = IPtr(unsafe_from_address=idx_addr)
    var dist = Ptr(unsafe_from_address=dist_addr)
    if m >= 32 and n * d >= 4096:
        @__parameter
        def work(q: Int):
            linear_query(train, query, idx, dist, n, d, k, metric, q)
        map[work](m)
    else:
        for q in range(m):
            linear_query(train, query, idx, dist, n, d, k, metric, q)


@export("mpf_knn_kdtree")
def mpf_knn_kdtree(
    train_addr: Int, query_addr: Int, dim_addr: Int, split_addr: Int,
    left_addr: Int, right_addr: Int, start_addr: Int, end_addr: Int,
    perm_addr: Int, roots_addr: Int, idx_addr: Int, dist_addr: Int,
    seen_addr: Int, stack_node_addr: Int, stack_bound_addr: Int,
    n: Int, d: Int, m: Int, k: Int, trees: Int, checks: Int, metric: Int,
) abi("C"):
    var train = Ptr(unsafe_from_address=train_addr)
    var query = Ptr(unsafe_from_address=query_addr)
    var node_dim = IPtr(unsafe_from_address=dim_addr)
    var split = Ptr(unsafe_from_address=split_addr)
    var left = IPtr(unsafe_from_address=left_addr)
    var right = IPtr(unsafe_from_address=right_addr)
    var start = IPtr(unsafe_from_address=start_addr)
    var end = IPtr(unsafe_from_address=end_addr)
    var perm = IPtr(unsafe_from_address=perm_addr)
    var roots = IPtr(unsafe_from_address=roots_addr)
    var idx = IPtr(unsafe_from_address=idx_addr)
    var dist = Ptr(unsafe_from_address=dist_addr)
    var seen = BPtr(unsafe_from_address=seen_addr)
    var stack_node = IPtr(unsafe_from_address=stack_node_addr)
    var stack_bound = Ptr(unsafe_from_address=stack_bound_addr)
    for q in range(m):
        var base = q * k
        for s in range(k):
            idx[unsafe_offset=base + s] = -1
            dist[unsafe_offset=base + s] = 1.7976931348623157e308
        for point in range(n):
            seen[unsafe_offset=point] = 0
        var stack_size = 0
        for tree in range(trees):
            stack_node[unsafe_offset=stack_size] = roots[unsafe_offset=tree]
            stack_bound[unsafe_offset=stack_size] = 0.0
            stack_size += 1
        var leaves = 0
        while stack_size > 0 and (checks < 0 or leaves < checks):
            var pick = 0
            for s in range(1, stack_size):
                if stack_bound[unsafe_offset=s] < stack_bound[unsafe_offset=pick]:
                    pick = s
            var node = Int(stack_node[unsafe_offset=pick])
            var node_bound = stack_bound[unsafe_offset=pick]
            stack_size -= 1
            stack_node[unsafe_offset=pick] = stack_node[unsafe_offset=stack_size]
            stack_bound[unsafe_offset=pick] = stack_bound[unsafe_offset=stack_size]
            while node_dim[unsafe_offset=node] >= 0:
                var axis = Int(node_dim[unsafe_offset=node])
                var delta = query[unsafe_offset=q * d + axis] - split[unsafe_offset=node]
                var near = Int(left[unsafe_offset=node]) if delta <= 0.0 else Int(right[unsafe_offset=node])
                var far = Int(right[unsafe_offset=node]) if delta <= 0.0 else Int(left[unsafe_offset=node])
                var extra = delta * delta if metric == 0 else (-delta if delta < 0.0 else delta)
                stack_node[unsafe_offset=stack_size] = Int64(far)
                stack_bound[unsafe_offset=stack_size] = node_bound + extra
                stack_size += 1
                node = near
            leaves += 1
            for pos in range(Int(start[unsafe_offset=node]), Int(end[unsafe_offset=node])):
                var point = Int(perm[unsafe_offset=pos])
                if seen[unsafe_offset=point] == 0:
                    seen[unsafe_offset=point] = 1
                    insert_neighbour(idx, dist, base, k, point, distance(query.unsafe_offset(q * d), train.unsafe_offset(point * d), d, metric))


@export("mpf_radius_linear")
def mpf_radius_linear(
    train_addr: Int, query_addr: Int, idx_addr: Int, dist_addr: Int,
    n: Int, d: Int, radius: Float64, metric: Int, sorted: Int, limit: Int,
) abi("C") -> Int:
    var train = Ptr(unsafe_from_address=train_addr)
    var query = Ptr(unsafe_from_address=query_addr)
    var idx = IPtr(unsafe_from_address=idx_addr)
    var dist = Ptr(unsafe_from_address=dist_addr)
    var count = 0
    for row in range(n):
        var value = distance(query, train.unsafe_offset(row * d), d, metric)
        if value <= radius and (limit < 0 or count < limit):
            idx[unsafe_offset=count] = Int64(row)
            dist[unsafe_offset=count] = value
            count += 1
    if sorted != 0:
        for i in range(1, count):
            var value = dist[unsafe_offset=i]
            var point = idx[unsafe_offset=i]
            var j = i
            while j > 0 and dist[unsafe_offset=j - 1] > value:
                dist[unsafe_offset=j] = dist[unsafe_offset=j - 1]
                idx[unsafe_offset=j] = idx[unsafe_offset=j - 1]
                j -= 1
            dist[unsafe_offset=j] = value
            idx[unsafe_offset=j] = point
    return count
