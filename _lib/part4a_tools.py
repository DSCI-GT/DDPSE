"""Shared helpers for Topics 4.1 and 4.2 (working with real data; optimization in reduced spaces).

Two synthetic plant-historian logs built on the running electrolyzer example (``_lib/electrolyzer.py``):

* ``stack_log``  one stack (A) run at loads chosen by operators, not by a design of experiments, with
                  four sensors that all track the load and a noisy power meter (Topic 4.1, Topic 4.2 Sec. 1-2).
* ``fleet_log``  stacks A, B, C sharing the demand D = 1.2 at operator-chosen splits, all stacks on,
                  with a noisy reading of the total power (Topic 4.2, last section).

Both are generated from fixed seeds, so every number the chapters quote is reproducible. The logs are
synthetic on purpose: the true power curve is known, so we can say how far a data-driven answer is from
the truth.

Also: ``find_solver`` (the book's solver-lookup pattern) and ``specific_energy`` (per-unit power per
per-unit hydrogen, the quantity whose minimum is a stack's best-efficiency load).
"""
import os
import shutil

import numpy as np
import pandas as pd

from . import electrolyzer as el
from .solvers import get_solver as find_solver  # noqa: E402  (merged into _lib/solvers.py)

WATER_PER_H2 = 18.015 / 2.016          # kg water consumed per kg H2
O2_PER_H2 = 15.999 / 2.016             # kg O2 produced per kg H2
KWH_PER_KG_PER_UNIT = el.MW_PER_UNIT * 1e3 / el.KG_H_PER_UNIT   # 1 per-unit specific energy in kWh/kg


def specific_energy(x, cell="A"):
    """Per-unit specific energy P(x)/x of one stack (1 = the ideal 32.7 kWh/kg)."""
    x = np.asarray(x, float)
    return el.p_scurve(x, cell) / x


def stack_log(n=400, seed=0, noise=0.01):
    """Hourly historian log of stack A. Columns are in physical units except ``load``.

    load        operator setpoint, fraction of full load (known exactly; the decision variable)
    h2_kg_h     hydrogen flow meter (1 % noise)
    water_kg_h  feed-water flow meter (1 % noise)
    o2_kg_h     oxygen flow meter (1.5 % noise)
    temp_C      stack outlet temperature (50 C + 25 C x full load, 1.5 C noise)
    power_MW    stack power meter, true S-curve power times (1 + noise * N(0, 1))
    """
    rng = np.random.default_rng(seed)
    x = 0.15 + 0.85 * rng.beta(2.0, 2.0, n)          # operators avoid both extremes
    h2 = el.to_kg_h(x) * (1 + 0.010 * rng.standard_normal(n))
    water = el.to_kg_h(x) * WATER_PER_H2 * (1 + 0.010 * rng.standard_normal(n))
    o2 = el.to_kg_h(x) * O2_PER_H2 * (1 + 0.015 * rng.standard_normal(n))
    temp = 50 + 25 * x + 1.5 * rng.standard_normal(n)
    power = el.to_mw(el.p_scurve(x, "A")) * (1 + noise * rng.standard_normal(n))
    return pd.DataFrame({"load": x, "h2_kg_h": h2, "water_kg_h": water, "o2_kg_h": o2,
                         "temp_C": temp, "power_MW": power})


def fleet_log(n=60, seed=1, D=1.2, noise=0.005):
    """Historian log of the three-stack fleet at demand D, all stacks on (loads 0.25-0.55).

    Operators split D around an equal share with some scatter; the C load is whatever makes the sum D.
    ``power`` is the measured total S-curve power in per-unit, with relative noise ``noise``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < n:
        xa, xb = D / 3 + 0.08 * rng.standard_normal(2)
        xc = D - xa - xb
        if min(xa, xb, xc) >= 0.25 and max(xa, xb, xc) <= 0.55:
            rows.append((xa, xb, xc))
    X = np.array(rows)
    p = np.array([el.bb2(r[:2], D) for r in X]) * (1 + noise * rng.standard_normal(n))
    return pd.DataFrame({"xA": X[:, 0], "xB": X[:, 1], "xC": X[:, 2], "power": p})
