"""Shared helpers for Topics 3.1-3.4 (derivative-free optimization on the electrolyzer black boxes).

The black boxes themselves live in `electrolyzer.py` (bb1, bb2, bb3). This module adds:

* Counter            wraps a black box and counts (and records) its evaluations
* compass            compass search (coordinate positive basis +-e_i, halve the step on failure)
* bb3_penalty        BB3 with the demand balance penalized instead of eliminated (three loads)
* bb3_reference      the true optimum of BB3 at demand D, by brute force over on/off configurations
                     (a check value only; Topic 2.6 obtains the same numbers with branch-and-bound)
* preview_samples    the Latin-hypercube sample points of BB1 used by the quadratic preview in Topic 3.1
                     and reused by Topic 3.6
* lipschitz_bb1      a valid Lipschitz constant for BB1 (global, or over a sub-interval)
* direct_certified   DIRECT-style trisection on BB1 with a Lipschitz lower bound (Topic 3.3)
* cosine_measure     the cosine measure of a set of directions in 2-D (Topic 3.2)
"""
import itertools
import numpy as np
from scipy.stats import qmc

from . import electrolyzer as el


class Counter:
    """Count the evaluations of a black box and keep the history of values returned."""
    def __init__(self, f):
        self.f, self.n, self.history = f, 0, []

    def __call__(self, x):
        self.n += 1
        v = self.f(np.atleast_1d(x))
        self.history.append(v)
        return v


def compass(f, x0, lo, hi, step=0.25, tol=1e-6, maxf=4000):
    """Compass search: poll x +- step e_i in order, move to the first improvement, else halve the step."""
    x = np.array(x0, float); fx = f(x); n = len(x)
    dirs = np.vstack([np.eye(n), -np.eye(n)])
    while step > tol and f.n < maxf:
        for d in dirs:
            y = np.clip(x + step * d, lo, hi); fy = f(y)
            if fy < fx - 1e-12:
                x, fx = y, fy
                break
        else:
            step /= 2
    return x, fx


def bb3_penalty(x, D=1.2, mu=10.0):
    """BB3 with all three loads as decisions and the demand balance penalized: cost + mu |sum x - D|."""
    loads = [0.0 if v < el.X_MIN else float(v) for v in np.atleast_1d(x)[:3]]
    return el.fleet_cost(loads) + mu * abs(sum(loads) - D)


def bb3_reference(D, h=1e-4):
    """True optimum of the full MINLP at demand D by brute force over the seven on/off configurations.

    Returns (cost, configuration name). A check value for the tables, not a method we teach.
    """
    best = (np.inf, None)
    for k in (1, 2, 3):
        for on in itertools.combinations(el.NAMES, k):
            if k == 1:
                loads = [np.array([D])]
            elif k == 2:
                x1 = np.arange(el.X_MIN, 1 + h / 2, h); loads = [x1, D - x1]
            else:
                g = np.arange(el.X_MIN, 1 + 1e-3 / 2, 1e-3)
                x1, x2 = (a.ravel() for a in np.meshgrid(g, g)); loads = [x1, x2, D - x1 - x2]
            ok = np.all([(v >= el.X_MIN) & (v <= 1) for v in loads], axis=0)
            if not np.any(ok):
                continue
            cost = sum(el.F_ON + el.p_scurve(v, c) for v, c in zip(loads, on))
            cost = np.where(ok, cost, np.inf)
            if cost.min() < best[0]:
                best = (float(cost.min()), "+".join(on))
    return best


def preview_samples(n, seed=1, D=0.9):
    """Latin-hypercube samples of BB1's single load on [0, D], sorted (shared by Topics 3.1 and 3.6)."""
    return np.sort(qmc.LatinHypercube(d=1, seed=seed).random(n)[:, 0] * D)


# ---- Lipschitz constant and the certified DIRECT of Topic 3.3 ---------------------------------
_XS = np.linspace(0, 0.9, 90001)
_FX = np.array([el.bb1([x]) for x in _XS])
_DF = np.abs(np.gradient(_FX, _XS))


def lipschitz_bb1(a=0.0, b=0.9, margin=1.01):
    """max |f'| of BB1 over [a, b] (sampled on a fine grid) times a 1% safety margin."""
    i, j = np.searchsorted(_XS, a), np.searchsorted(_XS, b)
    return _DF[max(i - 1, 0):min(j + 2, len(_XS))].max() * margin


def direct_certified(*, local=False, tol=1e-3, maxeval=20000, D=0.9):
    """Center-sampled trisection on BB1 with Lipschitz lower bounds f(c) - L (b - a)/2.

    Splits every interval whose bound could still beat the incumbent and discards intervals proven
    worse. Returns (evaluations to reach the gap tol, history of (evals, incumbent, lower bound)).
    With local=True each interval uses the Lipschitz constant over that interval only.
    """
    f = lambda x: el.bb1([x])
    L_glob = lipschitz_bb1()
    Lof = (lambda a, b: lipschitz_bb1(a, b)) if local else (lambda a, b: L_glob)
    ivs = [(0.0, D, f(D / 2))]; nev = 1; best = ivs[0][2]; hist = []
    while nev < maxeval:
        lbs = [c - Lof(a, b) * (b - a) / 2 for a, b, c in ivs]
        lb = min(lbs); hist.append((nev, best, lb))
        if best - lb <= tol:
            return nev, hist
        new = []
        for (a, b, c), l in zip(ivs, lbs):
            if l > best - tol:
                new.append((a, b, c)); continue
            w = (b - a) / 3
            left, right = (a, a + w), (b - w, b)
            fl, fr = f(sum(left) / 2), f(sum(right) / 2); nev += 2
            best = min(best, fl, fr)
            new += [(left[0], left[1], fl), (a + w, b - w, c), (right[0], right[1], fr)]
        ivs = [iv for iv in new if iv[2] - Lof(iv[0], iv[1]) * (iv[1] - iv[0]) / 2 <= best]
    return None, hist


def cosine_measure(D, n_angles=360001):
    """Cosine measure of a set of 2-D directions (columns of D): min over unit v of max_d v.d/|d|."""
    D = np.asarray(D, float); U = D / np.linalg.norm(D, axis=0)
    th = np.linspace(0, 2 * np.pi, n_angles)
    V = np.vstack([np.cos(th), np.sin(th)])
    return float((V.T @ U).max(axis=1).min())
