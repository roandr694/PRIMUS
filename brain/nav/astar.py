"""A* on a 2-D cost grid (8-connected, octile heuristic), compiled with numba; path smoothing by line of sight.

    path = astar(cost, start_ij, goal_ij)       # cost: float32 [ny, nx], inf = forbidden, >= 1 elsewhere
    pts  = simplify(path, cost)                 # fewer waypoints where the straight segment stays cheap

Cost is per cell (multiplies the step length), so a cell of cost 3 is "three times as long" to cross.
"""
import heapq
import math

import numpy as np

try:
    from numba import njit
except ImportError:                                  # pragma: no cover - numba is in the sim environment
    njit = None

SQ2 = math.sqrt(2.0)


def _astar_py(cost, si, sj, gi, gj, max_expand):
    ny, nx = cost.shape
    g = np.full((ny, nx), np.inf)
    parent = np.full((ny, nx), -1, np.int64)
    closed = np.zeros((ny, nx), np.bool_)
    g[si, sj] = 0.0
    heap = [(0.0, si, sj)]
    di = (-1, -1, -1, 0, 0, 1, 1, 1)
    dj = (-1, 0, 1, -1, 1, -1, 0, 1)
    dl = (SQ2, 1.0, SQ2, 1.0, 1.0, SQ2, 1.0, SQ2)
    n = 0
    while heap:
        f, i, j = heapq.heappop(heap)
        if closed[i, j]:
            continue
        closed[i, j] = True
        n += 1
        if (i == gi and j == gj) or n > max_expand:
            break
        for k in range(8):
            a, b = i + di[k], j + dj[k]
            if a < 0 or a >= ny or b < 0 or b >= nx or closed[a, b]:
                continue
            c = cost[a, b]
            if not np.isfinite(c):
                continue
            if dl[k] > 1.0 and (not np.isfinite(cost[i, b]) or not np.isfinite(cost[a, j])):
                continue                             # no corner cutting
            ng = g[i, j] + dl[k] * 0.5 * (c + cost[i, j])
            if ng < g[a, b]:
                g[a, b] = ng
                parent[a, b] = i * nx + j
                ddi, ddj = abs(a - gi), abs(b - gj)
                h = (ddi + ddj) + (SQ2 - 2.0) * min(ddi, ddj)
                heapq.heappush(heap, (ng + h, a, b))
    return g, parent


if njit is not None:
    @njit(cache=True)
    def _astar_nb(cost, si, sj, gi, gj, max_expand):
        ny, nx = cost.shape
        g = np.full((ny, nx), np.inf)
        parent = np.full((ny, nx), -1, np.int64)
        closed = np.zeros((ny, nx), np.bool_)
        g[si, sj] = 0.0
        # binary heap of (f, idx) in arrays
        cap = 1 << 16
        hf = np.empty(cap, np.float64)
        hi = np.empty(cap, np.int64)
        size = 0
        hf[0], hi[0] = 0.0, si * nx + sj
        size = 1
        di = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
        dj = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
        sq2 = math.sqrt(2.0)
        dl = np.array([sq2, 1.0, sq2, 1.0, 1.0, sq2, 1.0, sq2])
        n = 0
        while size > 0:
            # pop min
            f0, idx = hf[0], hi[0]
            size -= 1
            if size > 0:
                hf[0], hi[0] = hf[size], hi[size]
                p = 0
                while True:
                    l, r = 2 * p + 1, 2 * p + 2
                    m = p
                    if l < size and hf[l] < hf[m]:
                        m = l
                    if r < size and hf[r] < hf[m]:
                        m = r
                    if m == p:
                        break
                    hf[p], hf[m] = hf[m], hf[p]
                    hi[p], hi[m] = hi[m], hi[p]
                    p = m
            i, j = idx // nx, idx % nx
            if closed[i, j]:
                continue
            closed[i, j] = True
            n += 1
            if (i == gi and j == gj) or n > max_expand:
                break
            for k in range(8):
                a, b = i + di[k], j + dj[k]
                if a < 0 or a >= ny or b < 0 or b >= nx or closed[a, b]:
                    continue
                c = cost[a, b]
                if not np.isfinite(c):
                    continue
                if dl[k] > 1.0 and (not np.isfinite(cost[i, b]) or not np.isfinite(cost[a, j])):
                    continue
                ng = g[i, j] + dl[k] * 0.5 * (c + cost[i, j])
                if ng < g[a, b]:
                    g[a, b] = ng
                    parent[a, b] = i * nx + j
                    ddi, ddj = abs(a - gi), abs(b - gj)
                    h = (ddi + ddj) + (sq2 - 2.0) * min(ddi, ddj)
                    if size >= cap:                       # grow the heap
                        cap *= 2
                        nf = np.empty(cap, np.float64)
                        ni = np.empty(cap, np.int64)
                        nf[:size] = hf[:size]
                        ni[:size] = hi[:size]
                        hf, hi = nf, ni
                    # push
                    p = size
                    hf[p], hi[p] = ng + h, a * nx + b
                    size += 1
                    while p > 0:
                        q = (p - 1) // 2
                        if hf[q] <= hf[p]:
                            break
                        hf[p], hf[q] = hf[q], hf[p]
                        hi[p], hi[q] = hi[q], hi[p]
                        p = q
        return g, parent


def astar(cost, start, goal, max_expand=4_000_000):
    """-> list of (i, j) from start to goal, or None. start / goal: (i, j)."""
    cost = np.ascontiguousarray(cost, np.float64)
    si, sj = int(start[0]), int(start[1])
    gi, gj = int(goal[0]), int(goal[1])
    ny, nx = cost.shape
    if not (0 <= si < ny and 0 <= sj < nx and 0 <= gi < ny and 0 <= gj < nx):
        return None
    if not np.isfinite(cost[gi, gj]) or not np.isfinite(cost[si, sj]):
        return None
    fn = _astar_nb if njit is not None else _astar_py
    g, parent = fn(cost, si, sj, gi, gj, max_expand)
    if not np.isfinite(g[gi, gj]):
        return None
    path = [(gi, gj)]
    idx = parent[gi, gj]
    while idx >= 0:
        path.append((int(idx // nx), int(idx % nx)))
        idx = parent[path[-1]]
    return path[::-1]


def line_cells(a, b):
    """Cells on the segment a -> b (supercover by dense sampling)."""
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 2
    t = np.linspace(0.0, 1.0, n)
    i = np.round(a[0] + (b[0] - a[0]) * t).astype(int)
    j = np.round(a[1] + (b[1] - a[1]) * t).astype(int)
    return i, j


def simplify(path, cost, tol=1.15):
    """Greedy line-of-sight shortcutting: keep a straight segment while its cells are finite and its mean cost
    stays within tol of the path's own cost over that stretch."""
    if path is None or len(path) < 3:
        return path
    out = [path[0]]
    k = 0
    while k < len(path) - 1:
        best = k + 1
        for m in range(len(path) - 1, k + 1, -1):
            i, j = line_cells(path[k], path[m])
            c = cost[i, j]
            if not np.all(np.isfinite(c)):
                continue
            pi = np.array([p[0] for p in path[k:m + 1]])
            pj = np.array([p[1] for p in path[k:m + 1]])
            if c.max() <= cost[pi, pj].max() * tol:
                best = m
                break
        out.append(path[best])
        k = best
    return out
