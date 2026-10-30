"""The book's running example: an electrolyzer fleet meeting a hydrogen demand.

Chapters import this module instead of copying the model. Everything is computed in per-unit
(dimensionless) numbers and converted for display, the way power engineers work:

* load      x = 1 is one stack's full-load hydrogen output, 200 kg H2/h;
* power     1 per-unit is the ideal (reversible) power at full load, 6.53 MW
              (200 kg/h = 27.56 mol/s, times 237.1 kJ/mol);
* demand    D = 1.2 means 240 kg H2/h.

Solving in per-unit keeps every number the book quotes reproducible: a local MINLP solver's answer can
change with the units of the objective, while the global answers do not (see Topic 2.6).

Loss models, one per stage of Topics 1.4 to 2.6:

* linear       P(x) = e x                                   (LP)
* ohmic        P(x) = x + r x^2                             (convex NLP)
* fixed charge P(x) = F + e x when on                       (MILP)
* S-curve      P(x) = x + a x/(x + eps) + r x^2             (non-convex NLP; the concave term is a
                                                              deliberately exaggerated Faraday-efficiency
                                                              effect, inspired by Ulleberg 2003 and
                                                              Baumhof et al. 2023)
* full MINLP   S-curve + on-cost F_ON + minimum load X_MIN + on/off binaries

Black boxes for Topics 3.1-3.6 (the solver sees only f): bb1 (two A stacks, D = 0.9), bb2 (A, B, C, D = 1.2, no
binaries), bb3 (the full MINLP; a load below X_MIN means the stack is off; infeasible -> PENALTY).
"""
import numpy as np

# ---- units -------------------------------------------------------------------------------------
DG0 = 237.13e3                         # J/mol H2, liquid water at 298.15 K
F_CONST = 96485.3321                   # C/mol
E0 = DG0 / (2 * F_CONST)               # 1.229 V
KG_H_PER_UNIT = 200.0                  # kg H2/h at x = 1
MOL_S_PER_UNIT = KG_H_PER_UNIT / 2.01588 / 3.6
MW_PER_UNIT = MOL_S_PER_UNIT * DG0 / 1e6   # 6.534 MW

def to_kg_h(x): return np.asarray(x) * KG_H_PER_UNIT
def to_mw(p): return np.asarray(p) * MW_PER_UNIT

# ---- parameters (dimensionless; settled 2026-10-08, regrounded 2026-10-29/30) -------------------
NAMES = ("A", "B", "C")
E_LINEAR = {"A": 1.20, "B": 1.45, "C": 1.70}             # LP and MILP stages
F_FIXED = {"A": 0.30, "B": 0.40, "C": 0.20}              # MILP fixed charge (= a, the sharp-switch limit)
R_OHMIC = {"A": 0.50, "B": 0.35, "C": 0.70}
A_PARASITIC = {"A": 0.30, "B": 0.40, "C": 0.20}
EPS = 0.05
F_ON = 0.12
X_MIN = 0.10
PENALTY = 10.0

# ---- power curves ------------------------------------------------------------------------------
def p_linear(x, cell): return E_LINEAR[cell] * x
def p_ohmic(x, cell): return x + R_OHMIC[cell] * x ** 2
def p_scurve(x, cell):
    x = np.asarray(x, float)
    return np.where(x > 0, x + A_PARASITIC[cell] * x / (x + EPS) + R_OHMIC[cell] * x ** 2, 0.0)

def fleet_cost(loads, cells=NAMES):
    """Full-MINLP cost of a dispatch: a stack below X_MIN is off; on stacks pay F_ON + P(x)."""
    return sum(F_ON + float(p_scurve(x, c)) for x, c in zip(loads, cells) if x >= X_MIN)

# ---- black boxes (Topics 3.1-3.6) --------------------------------------------------------------------
def bb1(x, D=0.9):
    """Two A stacks share D; one decision, the first stack's load."""
    x1 = float(np.clip(np.atleast_1d(x)[0], 0, D))
    return float(p_scurve(x1, "A") + p_scurve(D - x1, "A"))

def bb2(x, D=1.2):
    """A, B, C share D with no binaries; C takes the rest."""
    xa, xb = np.atleast_1d(x)[:2]; xc = D - xa - xb
    if xc < 0 or xc > 1: return PENALTY
    return float(p_scurve(xa, "A") + p_scurve(xb, "B") + p_scurve(xc, "C"))

def bb3(x, D=1.2):
    """The full MINLP as a black box. Loads of A and B below X_MIN mean off; C takes the rest."""
    xa, xb = (0.0 if v < X_MIN else float(v) for v in np.atleast_1d(x)[:2])
    xc = D - xa - xb
    if xc < 0 or xc > 1 or 0 < xc < X_MIN: return PENALTY
    return fleet_cost((xa, xb, xc))
