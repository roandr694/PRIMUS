"""Correlative scan matching against a likelihood field (Olson 2009 style, brute force on a coarse-to-fine window).

    pose, score = match(grid, pts_local, guess=(x, y, th), win_xy=0.3, win_th=0.12)

pts_local: scan points [N, 2] in the robot frame (x forward, y left, i.e. math convention at heading th).
Pose (x, y, th) is the robot in the map frame, th the math angle of its forward direction. The score is the mean
likelihood of the transformed points (0 .. 1); callers reject matches below ~0.25.
"""
import math

import numpy as np


def transform(pts, x, y, th):
    c, s = math.cos(th), math.sin(th)
    return pts @ np.array([[c, s], [-s, c]]) + np.array([x, y])


def _score_batch(lf, origin, res, pts, xs, ys, ths):
    """Mean likelihood (bilinear in the field) for every (x, y, th) candidate -> [nth, ny, nx]."""
    ny_, nx_ = lf.shape
    out = np.empty((len(ths), len(ys), len(xs)), np.float32)
    for k, th in enumerate(ths):
        c, s = math.cos(th), math.sin(th)
        rx = pts[:, 0] * c - pts[:, 1] * s                        # rotated points [N]
        ry = pts[:, 0] * s + pts[:, 1] * c
        u = (rx[None, :] + xs[:, None] - origin[0]) / res - 0.5    # [nx, N] continuous cell coordinates
        v = (ry[None, :] + ys[:, None] - origin[1]) / res - 0.5    # [ny, N]
        j0 = np.floor(u).astype(np.int64)
        i0 = np.floor(v).astype(np.int64)
        fu, fv = (u - j0).astype(np.float32), (v - i0).astype(np.float32)
        okj = (j0 >= 0) & (j0 < nx_ - 1)
        oki = (i0 >= 0) & (i0 < ny_ - 1)
        j0 = np.clip(j0, 0, nx_ - 2)
        i0 = np.clip(i0, 0, ny_ - 2)
        I, J = i0[:, None, :], j0[None, :, :]
        FU, FV = fu[None, :, :], fv[:, None, :]
        val = (lf[I, J] * (1 - FU) + lf[I, J + 1] * FU) * (1 - FV) + (lf[I + 1, J] * (1 - FU) + lf[I + 1, J + 1] * FU) * FV
        val = np.where(oki[:, None, :] & okj[None, :, :], val, 0.0)
        out[k] = val.mean(-1)
    return out


def score_at(grid, pts, pose, sigma=0.08, max_pts=180):
    """Mean likelihood of the scan at one pose."""
    lf = grid.likelihood_field(sigma)
    if len(pts) > max_pts:
        pts = pts[np.linspace(0, len(pts) - 1, max_pts).astype(int)]
    return float(_score_batch(lf, grid.origin, grid.res, pts, np.array([pose[0]]), np.array([pose[1]]),
                              np.array([pose[2]]))[0, 0, 0])


def match(grid, pts, guess, win_xy=0.3, win_th=0.12, sigma=0.08, max_pts=180, prior=0.02):
    lf = grid.likelihood_field(sigma)
    if len(pts) > max_pts:                                        # subsample the scan evenly
        pts = pts[np.linspace(0, len(pts) - 1, max_pts).astype(int)]
    x0, y0, th0 = guess
    best = (x0, y0, th0)
    # coarse -> fine: (xy step, th step, xy half window, th half window)
    for step_xy, step_th, hw_xy, hw_th in ((0.05, 0.02, win_xy, win_th), (0.0125, 0.005, 0.05, 0.02)):
        bx, by, bth = best
        xs = bx + np.arange(-hw_xy, hw_xy + 1e-9, step_xy)
        ys = by + np.arange(-hw_xy, hw_xy + 1e-9, step_xy)
        ths = bth + np.arange(-hw_th, hw_th + 1e-9, step_th)
        S = _score_batch(lf, grid.origin, grid.res, pts, xs, ys, ths)
        # small prior towards the guess keeps flat corridors from sliding
        dx = (xs[None, None, :] - x0) ** 2 + (ys[None, :, None] - y0) ** 2
        dth = (ths[:, None, None] - th0) ** 2
        S2 = S - prior * (dx / max(win_xy, 1e-3) ** 2 + dth / max(win_th, 1e-3) ** 2)
        k, i, j = np.unravel_index(np.argmax(S2), S2.shape)
        best = (float(xs[j]), float(ys[i]), float(ths[k]))
        score = float(S[k, i, j])
    return best, score


def global_search(grid, pts, free_mask_xy, th_step=math.radians(5), max_pts=90, top=5):
    """Kidnapped-robot relocalisation: score the scan at candidate positions (free cells, subsampled) and all
    headings; returns the best candidates [(score, x, y, th)] refined with `match`."""
    lf = grid.likelihood_field(0.12)
    if len(pts) > max_pts:
        pts = pts[np.linspace(0, len(pts) - 1, max_pts).astype(int)]
    cand = free_mask_xy
    ths = np.arange(0, 2 * math.pi, th_step)
    scores = np.zeros((len(cand), len(ths)), np.float32)
    ny_, nx_ = lf.shape
    for k, th in enumerate(ths):
        p = transform(pts, 0.0, 0.0, th)
        jj = np.floor((p[None, :, 0] + cand[:, None, 0] - grid.origin[0]) / grid.res).astype(np.int64)
        ii = np.floor((p[None, :, 1] + cand[:, None, 1] - grid.origin[1]) / grid.res).astype(np.int64)
        ok = (ii >= 0) & (ii < ny_) & (jj >= 0) & (jj < nx_)
        v = lf[np.clip(ii, 0, ny_ - 1), np.clip(jj, 0, nx_ - 1)]
        scores[:, k] = np.where(ok, v, 0.0).mean(1)
    flat = np.argsort(scores.ravel())[::-1][:top * 4]
    out = []
    for f in flat:
        c, k = divmod(int(f), len(ths))
        pose, s = match(grid, pts, (cand[c, 0], cand[c, 1], ths[k]), win_xy=0.15, win_th=th_step)
        out.append((s, *pose))
    out.sort(reverse=True)
    return out[:top]
