"""Shared helpers for Topics 2.4-2.6: mixed-integer formulations, branch-and-bound, and global bounds.

Everything is in per-unit, using the parameters of the book's running example (`_lib/electrolyzer.py`).
Two small branch-and-bound codes live here, written to be read rather than to be fast:

* `fc_bb`   LP-based branch-and-bound for the fixed-charge dispatch MILP (Topics 2.4 and 2.5). Each node's
            LP relaxation is solved by HiGHS through `scipy.optimize.linprog`. Every node is logged.
* `minlp_bb` branch-and-bound that branches on binaries *and* on continuous intervals (spatial branching)
            for the S-curve dispatch problem (Topic 2.6). Each node's bound is a convex (secant)
            relaxation solved exactly through its one-dimensional price dual.

`draw_tree` draws a logged tree. `find_solver` locates an external solver executable.

The code reproduces the course's verified prototypes (arc_bridge.py sections 4-5 and minlp_prototype.py).
"""
import heapq
import re
import itertools
import os
import shutil

import numpy as np
from scipy.optimize import linprog, minimize

from . import electrolyzer as el
from .solvers import find_executable as find_solver  # noqa: E402  (merged into _lib/solvers.py)

NAMES = list(el.NAMES)


# ----------------------------------------------------------------------------------------------------
# Solvers
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Fixed-charge dispatch MILP (Topics 2.4, 2.5)
#   min sum_i (F_i y_i + e_i x_i)  s.t.  sum_i x_i = D,  x_i <= M_i y_i,  0 <= x_i <= cap_i,  y_i in {0, 1}
# Variables are ordered (x_1..x_n, y_1..y_n).
# ----------------------------------------------------------------------------------------------------
def fc_lp(F, e, D, M, bounds):
    """LP relaxation of the fixed-charge model with variable bounds `bounds`. Returns (z, value) or (None, inf)."""
    n = len(F)
    c = np.concatenate([e, F])
    A = [[1] * n + [0] * n, [-1] * n + [0] * n]
    A += [[1 if j == i else 0 for j in range(n)] + [-M[i] if j == i else 0 for j in range(n)] for i in range(n)]
    b = [D, -D] + [0] * n
    r = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if r.status != 0:
        return None, np.inf
    return r.x + 0.0, r.fun          # + 0.0 turns HiGHS's -0.0 into 0.0


def fc_bb(F, e, D, M=None, cap=None, order="dfs", names=None, tol=1e-6):
    """Branch-and-bound for the fixed-charge model. Branches on the most fractional y; depth-first
    takes the y = 1 child first, breadth-first takes the y = 0 child first.

    Returns a dict with the incumbent value `best`, its point `z`, the node count, pruning statistics,
    and `log`, a list of node records (dicts) in the order the nodes were processed."""
    F, e = np.asarray(F, float), np.asarray(e, float)
    n = len(F)
    M = [1.0] * n if M is None else list(M)
    cap = [1.0] * n if cap is None else list(cap)
    names = names or NAMES[:n]
    base = [(0, cap[i]) for i in range(n)] + [(0, 1)] * n
    queue = [dict(bounds=base, edge="root", parent=None, depth=0)]
    best, zbest, log, k = np.inf, None, [], 0
    stats = dict(branched=0, integer=0, bound=0, infeasible=0)
    while queue:
        nd = queue.pop() if order == "dfs" else queue.pop(0)
        k += 1
        z, val = fc_lp(F, e, D, M, nd["bounds"])
        rec = dict(id=k, parent=nd["parent"], edge=nd["edge"], depth=nd["depth"], bound=val, z=z)
        if z is None:
            rec.update(status="infeasible", text="pruned: infeasible"); stats["infeasible"] += 1
        elif val >= best - tol:
            rec.update(status="bound", text="pruned: bound %.4f >= %.4f" % (val, best)); stats["bound"] += 1
        else:
            y = z[n:]
            frac = [(min(y[i], 1 - y[i]), i) for i in range(n) if 1e-6 < y[i] < 1 - 1e-6]
            if not frac:
                best, zbest = val, z.copy()
                rec.update(status="integer", text="integer: new incumbent %.4f" % val); stats["integer"] += 1
            else:
                _, i = max(frac)
                rec.update(status="branch", var=names[i],
                           text="bound %.4f, y_%s = %.3f -> branch" % (val, names[i], y[i]))
                stats["branched"] += 1
                kids = []
                for v in (0, 1):
                    b = list(nd["bounds"]); b[n + i] = (v, v)
                    kids.append(dict(bounds=b, edge="y_%s=%d" % (names[i], v), parent=k, depth=nd["depth"] + 1))
                queue.extend(kids)          # dfs pops from the end, so the y = 1 child is processed first
        log.append(rec)
    return dict(best=best, z=zbest, nodes=k, stats=stats, log=log, names=names, n=n)


def print_tree_log(res):
    for r in res["log"]:
        print("%snode %d [%s]: %s" % ("    " * r["depth"], r["id"], r["edge"], r["text"]))


def fc_enumerate(F, e, D, M=None, cap=None, names=None):
    """Solve the LP for every on/off configuration. Returns (cost, configuration, loads), sorted."""
    n = len(F)
    M = [1.0] * n if M is None else list(M)
    cap = [1.0] * n if cap is None else list(cap)
    names = names or NAMES[:n]
    out = []
    for y in itertools.product([0, 1], repeat=n):
        b = [(0, cap[i] if y[i] else 0) for i in range(n)] + [(y[i], y[i]) for i in range(n)]
        z, v = fc_lp(np.asarray(F, float), np.asarray(e, float), D, M, b)
        cfg = "+".join(names[i] for i in range(n) if y[i]) or "none"
        out.append((v, cfg, None if z is None else np.round(z[:n], 4)))
    return sorted(out, key=lambda t: t[0])


def config(z, n, names=NAMES):
    return "+".join(names[i] for i in range(n) if z[n + i] > 0.5)


# ----------------------------------------------------------------------------------------------------
# Tree drawing (both codes produce records with id, parent, edge, depth, status, and a short label)
# ----------------------------------------------------------------------------------------------------
STATUS_COLOR = {"branch": "#003057", "spatial": "#4B8B9B", "integer": "#377117", "incumbent": "#377117",
                "bound": "#8E8B76", "infeasible": "#B3262B", "tight": "#8E8B76"}


def draw_tree(records, ax, label=None, edge_label=None):
    """Draw a tree from node records. Leaves are spaced evenly; parents sit above the mean of their
    children. `label(rec)` returns the text printed below each node; `edge_label(rec)` the text on the
    edge into it (default: the branching condition; return "" to omit)."""
    kids = {r["id"]: [] for r in records}
    root = None
    for r in records:
        if r["parent"] is None:
            root = r["id"]
        else:
            kids[r["parent"]].append(r["id"])
    for k in kids:
        kids[k].sort(key=lambda c: [r for r in records if r["id"] == c][0]["edge"], reverse=True)
    rec = {r["id"]: r for r in records}
    pos, count = {}, [0]

    def place(k):
        if not kids[k]:
            pos[k] = count[0]; count[0] += 1
        else:
            for c in kids[k]:
                place(c)
            pos[k] = np.mean([pos[c] for c in kids[k]])
    place(root)
    for k, cs in kids.items():
        for c in cs:
            ax.plot([pos[k], pos[c]], [-rec[k]["depth"], -rec[c]["depth"]], color="0.7", lw=1, zorder=1)
            txt = rec[c]["edge"] if edge_label is None else edge_label(rec[c])
            if not txt:
                continue
            m = re.fullmatch(r"y_(\w+)=(\d)", txt)          # 'y_B=1' -> mathtext $y_B = 1$
            if m:
                txt = "$y_{%s} = %s$" % m.groups()
            ax.text(0.5 * (pos[k] + pos[c]), -0.5 * (rec[k]["depth"] + rec[c]["depth"]), txt,
                    fontsize=7.5, ha="center", va="center", color="0.3",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    for k, r in rec.items():
        ax.scatter([pos[k]], [-r["depth"]], s=110, color=STATUS_COLOR.get(r["status"], "k"), zorder=3)
        if label is not None:
            ax.annotate(label(r), (pos[k], -r["depth"]), xytext=(0, -11), textcoords="offset points",
                        ha="center", va="top", fontsize=7.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    depth = max(r["depth"] for r in records)
    ax.set_ylim(-depth - 0.8, 0.35)
    ax.set_xlim(-0.6, count[0] - 0.4)
    return pos


# ----------------------------------------------------------------------------------------------------
# The S-curve model (Topic 2.6). P(x) = x + a x/(x + eps) + r x^2 for a running stack.
# ----------------------------------------------------------------------------------------------------
EPS, F_ON, X_MIN = el.EPS, el.F_ON, el.X_MIN
CELLS = {c: (el.A_PARASITIC[c], el.R_OHMIC[c]) for c in NAMES}     # (a, r)


def g(x, a):
    """The concave part of the S-curve (the smoothed fixed charge)."""
    return a * x / (x + EPS)


def P(x, a, r):
    return x + g(x, a) + r * x ** 2


def secant(a, lo, hi):
    """Intercept and slope (c0, c1) of the secant of g over [lo, hi]: the convex envelope of the concave g."""
    if hi - lo < 1e-12:
        return g(lo, a), 0.0
    c1 = (g(hi, a) - g(lo, a)) / (hi - lo)
    return g(lo, a) - c1 * lo, c1


def max_underestimation(a, lo, hi, npts=2001):
    xs = np.linspace(lo, hi, npts)
    c0, c1 = secant(a, lo, hi)
    return float(np.max(g(xs, a) - (c0 + c1 * xs)))


class Node:
    """A node: a box [lo_i, hi_i] on each load, and the status of each binary (None = free, 0 or 1)."""
    def __init__(self, lo, hi, ystat, depth, parent, edge):
        self.lo, self.hi, self.ystat = list(lo), list(hi), list(ystat)
        self.depth, self.parent, self.edge = depth, parent, edge
        self.id = None


def _eff_bounds(node, binaries):
    lo, hi = [], []
    for i in range(len(node.lo)):
        l, h = node.lo[i], node.hi[i]
        if binaries and node.ystat[i] == 0:
            l, h = 0.0, 0.0
        if binaries and node.ystat[i] == 1:
            l = max(l, X_MIN)
        lo.append(l); hi.append(h)
    return lo, hi


def relaxation_dual(node, cells, D, binaries):
    """The node's convex relaxation, solved exactly by bisection on the demand multiplier lambda.

    With g replaced by its secant c0 + c1 x on the node's box, and a free binary relaxed to y_i = x_i
    (optimal, since F_ON > 0), each stack's cost is k_i x_i + c0_i + r_i x_i^2 with k_i = 1 + c1_i (+ F_ON
    if its binary is free). The Lagrangian separates, so stack i's best response to a price lambda is
    x_i = clip((lambda - k_i) / (2 r_i), lo_i, hi_i). Bisection on sum_i x_i = D finds the KKT point;
    strong duality certifies the value as a lower bound. Returns (x, y, value, lambda, secants) or None."""
    n = len(cells)
    lo, hi = _eff_bounds(node, binaries)
    if sum(lo) > D + 1e-9 or sum(hi) < D - 1e-9:
        return None
    sec = [secant(cells[i][0], lo[i], hi[i]) for i in range(n)]
    k = np.array([1.0 + sec[i][1] + (F_ON if (binaries and node.ystat[i] is None) else 0.0) for i in range(n)])
    r = np.array([cells[i][1] for i in range(n)])
    lo_a, hi_a = np.array(lo), np.array(hi)

    def x_of(lam):
        return np.clip((lam - k) / (2 * r), lo_a, hi_a)
    a_, b_ = float(np.min(k - 2 * r) - 1.0), float(np.max(k + 2 * r) + 1.0)
    for _ in range(200):
        m = 0.5 * (a_ + b_)
        if np.sum(x_of(m)) < D:
            a_ = m
        else:
            b_ = m
    lam = 0.5 * (a_ + b_)
    x = x_of(lam)
    if abs(np.sum(x) - D) > 1e-10:
        x = np.clip(x + (D - np.sum(x)) / n, lo_a, hi_a)
    val = float(np.sum(k * x + np.array([s[0] for s in sec]) + r * x ** 2))
    if binaries:
        val += F_ON * sum(1.0 for i in range(n) if node.ystat[i] == 1)
        y = np.array([x[i] if node.ystat[i] is None else float(node.ystat[i]) for i in range(n)])
    else:
        y = np.ones(n)
    return x, y, val, lam, sec


def relaxation_local(node, cells, D, binaries):
    """The node problem with the TRUE (non-convex) curve, solved locally by SLSQP from an equal split.
    Its value is NOT a valid bound; Topic 2.6 uses it to show what goes wrong."""
    n = len(cells)
    lo, hi = _eff_bounds(node, binaries)
    if sum(lo) > D + 1e-9 or sum(hi) < D - 1e-9:
        return None
    nv = 2 * n if binaries else n

    def obj(v):
        val = sum(v[i] + g(v[i], cells[i][0]) + cells[i][1] * v[i] ** 2 for i in range(n))
        return val + (F_ON * np.sum(v[n:]) if binaries else 0.0)
    cons = [{"type": "eq", "fun": lambda v: np.sum(v[:n]) - D}]
    bnds = [(lo[i], hi[i]) for i in range(n)]
    if binaries:
        for i in range(n):
            st = node.ystat[i]
            bnds.append((0.0, 1.0) if st is None else (float(st), float(st)))
            cons.append({"type": "ineq", "fun": lambda v, i=i: v[i] - X_MIN * v[n + i]})
            cons.append({"type": "ineq", "fun": lambda v, i=i: v[n + i] - v[i]})
    xs = np.clip(np.full(n, D / n), lo, hi)
    s = np.concatenate([xs, np.clip(xs, 0, 1)]) if binaries else xs
    res = minimize(obj, s, method="SLSQP", bounds=bnds, constraints=cons, options={"ftol": 1e-12, "maxiter": 500})
    if not res.success or abs(np.sum(res.x[:n]) - D) > 1e-6:
        return None
    x = res.x[:n]
    y = res.x[n:] if binaries else np.ones(n)
    sec = [secant(cells[i][0], lo[i], hi[i]) for i in range(n)]
    return x, y, float(res.fun), None, sec


def true_cost(x, y, cells, binaries=True):
    return sum((F_ON * y[i] if binaries else 0.0) + (P(x[i], *cells[i]) if y[i] else 0.0) for i in range(len(cells)))


def _polish(x, y, node, cells, D, binaries):
    """Local solve of the TRUE problem with the binaries fixed, inside the node's box: an upper bound."""
    n = len(cells)
    lo, hi = _eff_bounds(node, binaries)
    for i in range(n):
        if not y[i]:
            lo[i], hi[i] = 0.0, 0.0
        elif binaries:
            lo[i] = max(lo[i], X_MIN)
    if sum(lo) > D + 1e-9 or sum(hi) < D - 1e-9:
        return None
    obj = lambda v: sum(P(v[i], *cells[i]) for i in range(n) if y[i])
    res = minimize(obj, np.clip(x, lo, hi), method="SLSQP", bounds=list(zip(lo, hi)),
                   constraints=[{"type": "eq", "fun": lambda v: np.sum(v) - D}], options={"ftol": 1e-12, "maxiter": 500})
    if not res.success or abs(np.sum(res.x) - D) > 1e-6:
        return None
    xx = res.x
    if binaries and any(y[i] and xx[i] < X_MIN - 1e-7 for i in range(n)):
        return None
    assert all(xx[i] <= 1e-9 for i in range(n) if not y[i]), "an off stack carries load"
    return xx, true_cost(xx, y, cells, binaries)


def minlp_bb(cells, D, binaries=True, bounds="convex", tol=1e-4, names=None, max_nodes=5000):
    """Best-first branch-and-bound on binaries and on continuous intervals.

    bounds='convex': each node's bound is the secant relaxation (valid). bounds='local': each node's
    'bound' is a local solve of the true non-convex node problem (not valid; for the Topic 2.6 demo).
    Returns a dict: incumbent x, y, value ub, final lower bound lb, stats, the node log and the
    (nodes, lower bound, upper bound) history."""
    nm = names or NAMES[:len(cells)]
    n = len(cells)
    solve = relaxation_dual if bounds == "convex" else relaxation_local
    root = Node([0.0] * n, [1.0] * n, [None] * n if binaries else [1] * n, 0, None, "root")
    heap, counter = [], itertools.count()
    inc_x, inc_y, ub = None, None, np.inf
    log, history, ids = [], [], itertools.count(1)
    stats = dict(nodes=0, binary_branches=0, spatial_branches=0, pruned_bound=0, pruned_infeasible=0, tight=0)
    rel = solve(root, cells, D, binaries)
    root.id = next(ids)
    heapq.heappush(heap, (rel[2], next(counter), root, rel))

    def record(node, lb, status, text):
        log.append(dict(id=node.id, parent=None if node.parent is None else node.parent.id, edge=node.edge,
                        depth=node.depth, bound=lb, status=status, text=text))

    while heap and stats["nodes"] < max_nodes:
        lb, _, node, rel = heapq.heappop(heap)
        stats["nodes"] += 1
        history.append((stats["nodes"], min([lb] + [h[0] for h in heap] + [ub]), ub))
        if lb >= ub - tol:
            stats["pruned_bound"] += 1; record(node, lb, "bound", "pruned: bound"); continue
        x, y, val, lam, sec = rel
        frac = [i for i in range(n) if binaries and node.ystat[i] is None and 1e-6 < y[i] < 1 - 1e-6]
        new_inc = False
        if not frac:
            yint = [int(round(t)) for t in y] if binaries else [1] * n
            pol = _polish(x, yint, node, cells, D, binaries)
            if pol is not None and pol[1] < ub - 1e-12:
                inc_x, inc_y, ub = pol[0], yint, pol[1]; new_inc = True
        if frac:
            i = max(frac, key=lambda k: min(y[k], 1 - y[k]))
            stats["binary_branches"] += 1
            kids = []
            for v in (0, 1):
                c = Node(node.lo, node.hi, node.ystat, node.depth + 1, node, "y_%s=%d" % (nm[i], v))
                c.ystat[i] = v; kids.append(c)
            record(node, lb, "branch", "branch on y_%s (= %.3f)" % (nm[i], y[i]))
        else:
            gaps = [(g(x[i], cells[i][0]) - (sec[i][0] + sec[i][1] * x[i])) if (not binaries or round(y[i]) == 1) else 0.0
                    for i in range(n)]
            i = int(np.argmax(gaps))
            if bounds == "local" or gaps[i] <= tol / n:
                stats["tight"] += 1
                record(node, lb, "incumbent" if new_inc else "tight", "relaxation tight: node solved"); continue
            lo_i, hi_i = (b[i] for b in _eff_bounds(node, binaries))
            cut = x[i] if (lo_i + 1e-3 < x[i] < hi_i - 1e-3) else 0.5 * (lo_i + hi_i)
            stats["spatial_branches"] += 1
            left = Node(node.lo, node.hi, node.ystat, node.depth + 1, node, "x_%s<=%.2f" % (nm[i], cut))
            right = Node(node.lo, node.hi, node.ystat, node.depth + 1, node, "x_%s>=%.2f" % (nm[i], cut))
            left.hi[i] = cut; right.lo[i] = cut
            kids = [left, right]
            record(node, lb, "incumbent" if new_inc else "spatial",
                   "split x_%s at %.3f (secant gap %.4f)%s" % (nm[i], cut, gaps[i], ", new incumbent %.5f" % ub if new_inc else ""))
        for c in kids:
            c.id = next(ids)
            r_ = solve(c, cells, D, binaries)
            if r_ is None:
                stats["pruned_infeasible"] += 1; record(c, None, "infeasible", "pruned: infeasible"); continue
            if r_[2] >= ub - tol:
                stats["pruned_bound"] += 1; record(c, r_[2], "bound", "pruned: bound"); continue
            heapq.heappush(heap, (r_[2], next(counter), c, r_))
    final_lb = min([h[0] for h in heap] + [ub]) if heap else ub
    history.append((stats["nodes"], final_lb, ub))
    return dict(x=inc_x, y=inc_y, ub=ub, lb=final_lb, stats=stats, log=log, history=history, names=nm)


def minlp_config(res):
    return "+".join(res["names"][i] for i in range(len(res["y"])) if res["y"][i])


def brute_force(cells, D, binaries=True, h=0.0005, names=None):
    """Ground truth: enumerate configurations; scan the split on a fine grid. Returns sorted
    (cost, configuration, loads dict)."""
    nm = names or NAMES[:len(cells)]
    n = len(cells)
    out = []
    for y in (itertools.product([0, 1], repeat=n) if binaries else [(1,) * n]):
        on = [i for i in range(n) if y[i]]
        lo_on = X_MIN if binaries else 0.0
        if not on or len(on) < D - 1e-9 or len(on) * lo_on > D + 1e-9:
            continue
        c0 = F_ON * len(on) if binaries else 0.0
        if len(on) == 1:
            if lo_on <= D <= 1:
                out.append((c0 + P(D, *cells[on[0]]), "+".join(nm[i] for i in on), {nm[on[0]]: D}))
        elif len(on) == 2:
            g1 = np.arange(lo_on, 1 + h / 2, h); g2 = D - g1
            ok = (g2 >= lo_on - 1e-12) & (g2 <= 1 + 1e-12)
            v = np.where(ok, c0 + P(g1, *cells[on[0]]) + P(g2, *cells[on[1]]), np.inf)
            k = int(np.argmin(v))
            if np.isfinite(v[k]):
                out.append((v[k], "+".join(nm[i] for i in on), {nm[on[0]]: g1[k], nm[on[1]]: g2[k]}))
        else:
            gg = np.arange(lo_on, 1 + h / 2, 2 * h)
            X1, X2 = np.meshgrid(gg, gg); X3 = D - X1 - X2
            ok = (X3 >= lo_on - 1e-12) & (X3 <= 1 + 1e-12)
            v = np.where(ok, c0 + P(X1, *cells[0]) + P(X2, *cells[1]) + P(X3, *cells[2]), np.inf)
            k = np.unravel_index(np.argmin(v), v.shape)
            if np.isfinite(v[k]):
                out.append((v[k], "+".join(nm), {nm[0]: X1[k], nm[1]: X2[k], nm[2]: X3[k]}))
    return sorted(out, key=lambda t: t[0])
