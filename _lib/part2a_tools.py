"""Shared helpers for Topics 2.1-2.3: nonlinear programs, Lagrange multipliers and KKT, duality.

Everything is in per-unit for the fleet (see `_lib/electrolyzer.py`) and in SI units (J, mol, V) for the
electrochemical kernel. Topic 2.2 writes the kernel model out in full; `kernel_nlp` here is the same model,
kept so that Topic 2.3 can reuse it.

* `find_solver`, `get_solver`  locate an external solver (PATH, then the IDAES binaries), or return None.
* `ohmic_dispatch`             the ohmic dispatch  min sum(x + r x^2)  s.t. sum x = D, 0 <= x <= cap,
                               solved exactly from its KKT conditions by bisection on the price lambda.
* `ohmic_dual`                 the Lagrangian dual function of the same problem (Topic 2.3).
* `kernel_nlp`                 the closed-cell electrolysis kernel (Topic 2.2) in Pyomo, solved by Ipopt.
"""
import os
import shutil

import numpy as np

from . import electrolyzer as el
from .solvers import find_executable as find_solver  # noqa: E402  (merged into _lib/solvers.py)
from .solvers import get_solver as _get_solver  # noqa: E402


def get_solver(name):
    """Return a working Pyomo solver for `name`, or None (with a message) if it is not installed."""
    return _get_solver(name, message=True)



# ---- physical constants for the kernel ---------------------------------------------------------
F = el.F_CONST                 # C/mol
R = 8.314462618                # J/(mol K)
T = 298.15                     # K
P0 = 1e5                       # Pa, 1 bar reference
DG0 = el.DG0                   # J/mol H2, H2O(l) -> H2 + 1/2 O2
E0 = el.E0                     # V
V_HEAD = 1e-3                  # m^3, 1 L headspace
N_WATER = 1000.0               # mol of liquid water, enough to stay in excess over the sweeps


# ---- ohmic dispatch --------------------------------------------------------------------------
def _r_vec(r):
    if r is None:
        return np.array([el.R_OHMIC[c] for c in el.NAMES])
    return np.asarray(r, float)


def ohmic_loads(lam, r=None, cap=1.0):
    """Each stack's best load at price lam: the inner problem min x + r x^2 - lam x on [0, cap]."""
    r = _r_vec(r)
    return np.clip((lam - 1.0) / (2 * r), 0.0, cap)


def ohmic_dispatch(D, r=None, cap=1.0, tol=1e-13):
    """Solve min sum(x_i + r_i x_i^2) s.t. sum x_i = D, 0 <= x_i <= cap from the KKT conditions.

    Returns (lam, x, cost, mu_cap): the demand multiplier (the hydrogen price), the loads, the total
    power, and each stack's capacity multiplier lam - (1 + 2 r_i x_i) (zero unless the stack is full).
    """
    r = _r_vec(r)
    if D > cap * len(r) + 1e-12:
        raise ValueError("demand exceeds fleet capacity")
    lo, hi = 1.0, 1.0 + 2 * r.max() * cap + 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if ohmic_loads(mid, r, cap).sum() < D:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    x = ohmic_loads(lam, r, cap)
    cost = float(np.sum(x + r * x ** 2))
    mu_cap = np.where(x >= cap - 1e-9, lam - (1 + 2 * r * x), 0.0)
    return lam, x, cost, mu_cap


def ohmic_dual(lam, D, r=None, cap=1.0):
    """Dual function g(lam) = lam D + sum_i min_{0<=x<=cap} (x + r_i x^2 - lam x)."""
    r = _r_vec(r)
    lam = np.atleast_1d(np.asarray(lam, float))
    out = []
    for L in lam:
        x = ohmic_loads(L, r, cap)
        out.append(L * D + np.sum(x + r * x ** 2 - L * x))
    out = np.array(out)
    return out if out.size > 1 else float(out[0])


# ---- the electrochemical kernel -----------------------------------------------------------------
def kernel_nlp(U, bH=2 * N_WATER, bO=N_WATER, V=V_HEAD, solver=None):
    """Closed-cell electrolysis at voltage U (Topic 2.2): min G - F U q s.t. H, O and charge balances.

    Returns the solved Pyomo model with duals (m.dual), or None if Ipopt is not available.
    """
    import pyomo.environ as pyo
    opt = solver or get_solver("ipopt")
    if opt is None:
        return None
    m = pyo.ConcreteModel()
    m.nw = pyo.Var(bounds=(1e-12, None), initialize=N_WATER)
    m.nh = pyo.Var(bounds=(1e-14, None), initialize=1e-2)
    m.no = pyo.Var(bounds=(1e-14, None), initialize=5e-4)
    m.q = pyo.Var(bounds=(0, None), initialize=2e-3)
    gas = lambda n: n * (pyo.log(n * R * T / (V * P0)) - 1) * R * T
    m.obj = pyo.Objective(expr=-DG0 * m.nw + gas(m.nh) + gas(m.no) - F * U * m.q)
    m.H = pyo.Constraint(expr=2 * m.nw + 2 * m.nh == bH)
    m.O = pyo.Constraint(expr=m.nw + 2 * m.no == bO)
    m.e = pyo.Constraint(expr=m.q - 2 * m.nh == 0)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    opt.solve(m, options={"tol": 1e-12, "acceptable_tol": 1e-12})
    return m


def kernel_closed_form(U, V=V_HEAD):
    """Exact closed-cell equilibrium with water in excess: Nernst gives n_H2; returns (n_H2, n_O2, lam_H, lam_O).

    p_H2 * sqrt(p_O2) = exp(2F(U - E0)/RT) with p_O2 = p_H2 / 2. Multipliers from the identities
    mu_O2 = 2 lam_O and mu_H2O = 2 lam_H + lam_O. Used as the fallback when Ipopt is missing.
    """
    K = np.exp(2 * F * (U - E0) / (R * T))
    pH = (K * np.sqrt(2)) ** (2 / 3)          # bar; p_H2^{3/2} / sqrt(2) = K
    pO = pH / 2
    nH, nO = pH * P0 * V / (R * T), pO * P0 * V / (R * T)
    lam_O = 0.5 * R * T * np.log(pO)
    lam_H = 0.5 * (-DG0 - lam_O)
    return nH, nO, lam_H, lam_O
