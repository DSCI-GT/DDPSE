"""Shared helpers for Topics 4.3, 4.4, 5.1 and 5.2.

* ``find_solver``      locate a Pyomo solver (PATH, then the IDAES binaries), or return None;
* ``mlp``              a small fully connected PyTorch network;
* ``train``            a plain full-batch Adam loop that records the loss history;
* ``omlt_network``     turn a trained ``mlp`` into an OMLT ``NetworkDefinition`` (no ONNX needed);
* ``Reactor``          the series reaction A -> B -> C used in Topics 4.3 and 4.4.

Nothing here is required reading; the chapters explain every piece they use.
"""
import os
import shutil

import numpy as np
from .solvers import get_solver as find_solver  # noqa: E402  (merged into _lib/solvers.py)



# ---- solvers -----------------------------------------------------------------------------------
# ---- neural networks ---------------------------------------------------------------------------
def mlp(n_in, n_out, width=16, depth=2, seed=0):
    """A tanh multilayer perceptron with ``depth`` hidden layers of ``width`` nodes."""
    import torch
    torch.manual_seed(seed)
    layers, n = [], n_in
    for _ in range(depth):
        layers += [torch.nn.Linear(n, width), torch.nn.Tanh()]
        n = width
    layers.append(torch.nn.Linear(n, n_out))
    return torch.nn.Sequential(*layers).double()


def train(params, loss_fn, steps=2000, lr=1e-2, record=1):
    """Minimize ``loss_fn()`` over ``params`` with Adam. Returns the loss history."""
    import torch
    opt = torch.optim.Adam(params, lr=lr)
    hist = []
    for k in range(steps):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        if k % record == 0:
            hist.append(loss.item())
    return np.array(hist)


def omlt_network(net, x_offset, x_factor, y_offset, y_factor, input_bounds):
    """Build an OMLT NetworkDefinition from a tanh ``mlp``.

    The network was trained on scaled inputs (x - x_offset)/x_factor and predicts scaled outputs
    (y - y_offset)/y_factor. ``input_bounds`` are bounds on the *unscaled* inputs.
    """
    import torch
    from omlt import OffsetScaling
    from omlt.neuralnet import NetworkDefinition
    from omlt.neuralnet.layer import DenseLayer, InputLayer

    x_offset, x_factor = np.atleast_1d(x_offset), np.atleast_1d(x_factor)
    y_offset, y_factor = np.atleast_1d(y_offset), np.atleast_1d(y_factor)
    scaler = OffsetScaling(
        offset_inputs=list(x_offset), factor_inputs=list(x_factor),
        offset_outputs=list(y_offset), factor_outputs=list(y_factor))
    scaled_bounds = {i: ((lo - x_offset[i]) / x_factor[i], (hi - x_offset[i]) / x_factor[i])
                     for i, (lo, hi) in enumerate(input_bounds)}
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    nd = NetworkDefinition(scaling_object=scaler, scaled_input_bounds=scaled_bounds)
    prev = InputLayer([lin[0].in_features])
    nd.add_layer(prev)
    for k, L in enumerate(lin):
        W = L.weight.detach().numpy().T          # OMLT wants (n_in, n_out)
        b = L.bias.detach().numpy()
        act = "linear" if k == len(lin) - 1 else "tanh"
        layer = DenseLayer(prev.output_size, [W.shape[1]], activation=act, weights=W, biases=b)
        nd.add_layer(layer)
        nd.add_edge(prev, layer)
        prev = layer
    return nd


# ---- the reactor of Topics 4.3 and 4.4 ---------------------------------------------------------
class Reactor:
    """Isothermal plug-flow (or batch) reactor, A -> B -> C, at temperature T (K).

    True kinetics (the "plant"): r1 = k1(T) C_A / (1 + K C_A), r2 = k2(T) C_B, Arrhenius k's.
    Concentrations in mol/L, residence time tau in min.
    """
    R = 8.314
    T_REF = 350.0
    K1_REF, E1 = 0.60, 50e3       # 1/min at T_REF, J/mol
    K2_REF, E2 = 0.15, 90e3
    K_ADS = 2.0                   # L/mol, inhibition by A
    CA0 = 1.0

    @classmethod
    def k1(cls, T): return cls.K1_REF * np.exp(-cls.E1 / cls.R * (1 / np.asarray(T) - 1 / cls.T_REF))

    @classmethod
    def k2(cls, T): return cls.K2_REF * np.exp(-cls.E2 / cls.R * (1 / np.asarray(T) - 1 / cls.T_REF))

    @classmethod
    def r1_true(cls, CA, T): return cls.k1(T) * CA / (1 + cls.K_ADS * CA)

    @classmethod
    def simulate(cls, T, tau, r1=None, n=400):
        """Integrate to each tau in ``tau`` (array) with RK4; returns (C_A, C_B) arrays."""
        r1 = cls.r1_true if r1 is None else r1
        tau = np.atleast_1d(np.asarray(tau, float))
        k2 = cls.k2(T)

        def f(c):
            ra = r1(c[0], T)
            return np.array([-ra, ra - k2 * c[1]])

        out = []
        for t in tau:
            h, c = t / n, np.array([cls.CA0, 0.0])
            for _ in range(n):
                a = f(c); b = f(c + h / 2 * a); d = f(c + h / 2 * b); e = f(c + h * d)
                c = c + h / 6 * (a + 2 * b + 2 * d + e)
            out.append(c)
        out = np.array(out)
        return out[:, 0], out[:, 1]

    @classmethod
    def measure(cls, temps, taus, sigma, rng):
        """Plant data: rows of (T [K], tau [min], C_A, C_B [mol/L]) with Gaussian noise ``sigma``."""
        rows = []
        for T in temps:
            ca, cb = cls.simulate(T, taus, n=100)
            rows += [[T, t, a, b] for t, a, b in zip(taus, ca, cb)]
        rows = np.array(rows)
        rows[:, 2:] += rng.normal(0, sigma, rows[:, 2:].shape)
        return rows


def rk4_reactor_pyomo(rate_def, n_steps=16, T_bounds=(330, 360), tau_bounds=(0.5, 4), scale=0.3):
    """Pyomo model of the A -> B -> C reactor with a learned first-step rate (Topic 4.3).

    The mole balances are written as ``n_steps`` explicit RK4 steps of length tau/n_steps. At every
    stage of every step an OMLT block evaluates the network ``rate_def`` (inputs C_A and T), and the
    rate is r1 = scale * C_A * softplus(network output). Decision variables: m.T (K), m.tau (min).
    The objective, maximize the outlet C_B, is m.obj.
    """
    import pyomo.environ as pyo
    from omlt import OmltBlock
    from omlt.neuralnet import FullSpaceSmoothNNFormulation

    R = Reactor
    m = pyo.ConcreteModel()
    m.T = pyo.Var(bounds=T_bounds, initialize=sum(T_bounds) / 2)
    m.tau = pyo.Var(bounds=tau_bounds, initialize=sum(tau_bounds) / 2)
    m.K, m.I, m.S = pyo.RangeSet(0, n_steps), pyo.RangeSet(0, n_steps - 1), pyo.RangeSet(1, 4)
    m.ca = pyo.Var(m.K, bounds=(0, 1), initialize=0.5)
    m.cb = pyo.Var(m.K, bounds=(0, 1), initialize=0.3)
    m.ca[0].fix(R.CA0)
    m.cb[0].fix(0.0)
    m.sa = pyo.Var(m.I, m.S, bounds=(0, 1), initialize=0.5)     # stage values of C_A
    m.sb = pyo.Var(m.I, m.S, initialize=0.3)                     # stage values of C_B
    m.nn = OmltBlock(m.I, m.S)
    for i in m.I:
        for s in m.S:
            m.nn[i, s].build_formulation(FullSpaceSmoothNNFormulation(rate_def))

    k2 = R.K2_REF * pyo.exp(-R.E2 / R.R * (1 / m.T - 1 / R.T_REF))
    r1 = lambda i, s: scale * m.sa[i, s] * pyo.log(1 + pyo.exp(m.nn[i, s].outputs[0]))
    h = m.tau / n_steps
    m.rk = pyo.ConstraintList()
    for i in m.I:
        for s in m.S:
            m.rk.add(m.nn[i, s].inputs[0] == m.sa[i, s])
            m.rk.add(m.nn[i, s].inputs[1] == m.T)
        m.rk.add(m.sa[i, 1] == m.ca[i])
        m.rk.add(m.sb[i, 1] == m.cb[i])
        for s, c in ((2, 0.5), (3, 0.5), (4, 1.0)):
            m.rk.add(m.sa[i, s] == m.ca[i] - c * h * r1(i, s - 1))
            m.rk.add(m.sb[i, s] == m.cb[i] + c * h * (r1(i, s - 1) - k2 * m.sb[i, s - 1]))
        fa = [-r1(i, s) for s in m.S]
        fb = [r1(i, s) - k2 * m.sb[i, s] for s in m.S]
        m.rk.add(m.ca[i + 1] == m.ca[i] + h / 6 * (fa[0] + 2 * fa[1] + 2 * fa[2] + fa[3]))
        m.rk.add(m.cb[i + 1] == m.cb[i] + h / 6 * (fb[0] + 2 * fb[1] + 2 * fb[2] + fb[3]))
    m.obj = pyo.Objective(expr=m.cb[n_steps], sense=pyo.maximize)
    return m


# ---- LP/MILP solve with a scipy fallback (Topics 5.1, 5.2) --------------------------------------
def solve_linear(model, prefer="appsi_highs"):
    """Solve a linear Pyomo model (LP or MILP) and load the solution into its variables.

    Uses HiGHS through Pyomo when it is available; otherwise compiles the model to matrix form and
    solves it with ``scipy.optimize.milp`` (which also calls HiGHS). Returns the objective value.
    """
    import pyomo.environ as pyo
    solver = find_solver(prefer)
    if solver is not None:
        res = solver.solve(model)
        if str(res.solver.termination_condition) not in ("optimal", "TerminationCondition.optimal"):
            raise RuntimeError(f"solver returned {res.solver.termination_condition}")
        return pyo.value(next(model.component_data_objects(pyo.Objective, active=True)))

    from scipy.optimize import Bounds, LinearConstraint, milp
    from pyomo.repn.plugins.standard_form import LinearStandardFormCompiler
    rep = LinearStandardFormCompiler().write(model, mixed_form=True, set_sense=pyo.minimize)
    A, rhs = rep.A.toarray(), np.array(rep.rhs, float)
    lo = np.array([-np.inf if r.bound_type == 1 else b for r, b in zip(rep.rows, rhs)])
    hi = np.array([np.inf if r.bound_type == -1 else b for r, b in zip(rep.rows, rhs)])
    cols = rep.columns
    lb = [-np.inf if v.lb is None else v.lb for v in cols]
    ub = [np.inf if v.ub is None else v.ub for v in cols]
    integ = [1 if v.is_integer() or v.is_binary() else 0 for v in cols]
    c = np.asarray(rep.c.toarray()).ravel()
    out = milp(c, constraints=LinearConstraint(A, lo, hi), bounds=Bounds(lb, ub), integrality=integ)
    if not out.success:
        raise RuntimeError(out.message)
    for v, val in zip(cols, out.x):
        v.set_value(float(val), skip_validation=True)
    return pyo.value(next(model.component_data_objects(pyo.Objective, active=True)))
