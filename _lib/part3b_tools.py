"""Spatial branch-and-bound on BB1's equations, for Topic 3.6's find-versus-prove comparison.

BB1 is two type-A stacks sharing D = 0.9 (see electrolyzer.bb1). The black-box certificate for the same
comparison (DIRECT with a Lipschitz constant) is part3a_tools.direct_certified, shared with Topic 3.3.

* spatial_bb_bb1   spatial branch-and-bound on the EQUATIONS (Topic 2.6). Each node is a box on
                   (x1, x2); the concave term a x/(x+eps) is replaced by its secant over the box, the
                   resulting separable convex problem is solved exactly by bisection on its demand
                   multiplier (the 1-D dual, Topic 2.3), and its value is a valid lower bound.

The history is a list of (relaxation solves so far, incumbent, global lower bound). Every node created is
one relaxation solve, so `solves` is also the number of nodes created; `nodes` counts nodes processed.
"""
import heapq
import itertools
import numpy as np
from scipy.optimize import minimize

from . import electrolyzer as el

A_, R_ = el.A_PARASITIC["A"], el.R_OHMIC["A"]
D1 = 0.9


def _g(x):                       # the concave part of the S-curve
    return A_ * x / (x + el.EPS)


def _P(x):
    return x + _g(x) + R_ * x ** 2


def _secant(lo, hi):
    if hi - lo < 1e-12:
        return _g(lo), 0.0
    c1 = (_g(hi) - _g(lo)) / (hi - lo)
    return _g(lo) - c1 * lo, c1


def _relax(lo, hi, D=D1):
    """Secant relaxation on box [lo, hi]; solved exactly by bisection on the demand multiplier."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if lo.sum() > D + 1e-9 or hi.sum() < D - 1e-9:
        return None
    sec = [_secant(l, h) for l, h in zip(lo, hi)]
    k = np.array([1.0 + s[1] for s in sec])
    x_of = lambda lam: np.clip((lam - k) / (2 * R_), lo, hi)
    a, b = float(np.min(k - 2 * R_) - 1), float(np.max(k + 2 * R_) + 1)
    for _ in range(200):
        m = 0.5 * (a + b)
        a, b = (m, b) if x_of(m).sum() < D else (a, m)
    x = x_of(0.5 * (a + b))
    if abs(x.sum() - D) > 1e-10:
        x = np.clip(x + (D - x.sum()) / 2, lo, hi)
    val = float(np.sum(k * x + np.array([s[0] for s in sec]) + R_ * x ** 2))
    return x, val, sec


def _polish(x, lo, hi, D=D1):
    """Local solve of the true problem inside the box: any feasible result is an upper bound."""
    res = minimize(lambda v: _P(v[0]) + _P(v[1]), np.clip(x, lo, hi), method="SLSQP",
                   bounds=list(zip(lo, hi)), constraints=[{"type": "eq", "fun": lambda v: v.sum() - D}],
                   options={"ftol": 1e-12, "maxiter": 500})
    if not res.success or abs(res.x.sum() - D) > 1e-6:
        return None
    return res.x, float(_P(res.x[0]) + _P(res.x[1]))


def spatial_bb_bb1(tol=1e-4, max_nodes=500):
    """Best-first spatial branch-and-bound on BB1's equations.

    Returns dict with x, ub, lb, nodes (nodes processed), solves (relaxation solves), spatial (branches),
    history [(relaxation solves so far, incumbent, global lower bound)]."""
    count = itertools.count()
    root_lo, root_hi = [0.0, 0.0], [1.0, 1.0]
    rel = _relax(root_lo, root_hi)
    solves, nodes, spatial = 1, 0, 0
    heap = [(rel[1], next(count), root_lo, root_hi, rel)]
    ub, xbest, hist = np.inf, None, []
    while heap and nodes < max_nodes:
        lb, _, lo, hi, (x, val, sec) = heapq.heappop(heap)
        nodes += 1
        if lb >= ub - tol:
            hist.append((solves, ub, min([lb] + [h[0] for h in heap] + [ub])))
            continue
        pol = _polish(x, lo, hi)
        if pol is not None and pol[1] < ub - 1e-12:
            xbest, ub = pol
        gaps = [_g(x[i]) - (sec[i][0] + sec[i][1] * x[i]) for i in range(2)]
        i = int(np.argmax(gaps))
        if gaps[i] > tol / 2:
            cut = x[i] if lo[i] + 1e-3 < x[i] < hi[i] - 1e-3 else 0.5 * (lo[i] + hi[i])
            spatial += 1
            for side in (0, 1):
                klo, khi = list(lo), list(hi)
                if side == 0:
                    khi[i] = cut
                else:
                    klo[i] = cut
                r = _relax(klo, khi)
                solves += 1
                if r is not None and r[1] < ub - tol:
                    heapq.heappush(heap, (r[1], next(count), klo, khi, r))
        glb = min([h[0] for h in heap] + [ub])
        hist.append((solves, ub, min(glb, ub)))
    lb_final = min([h[0] for h in heap] + [ub])
    return dict(x=xbest, ub=ub, lb=lb_final, nodes=nodes, solves=solves, spatial=spatial, history=hist)
