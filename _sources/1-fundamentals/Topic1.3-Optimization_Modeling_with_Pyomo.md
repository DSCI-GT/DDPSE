---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{contents}
:local:
:depth: 2
```

# 1.3 Optimization Modeling with Pyomo

:::{admonition} Draft status: first draft
:class: warning
First draft (2026-10-30), written from the course design notes. Not yet reviewed.
:::

```{admonition} Lectures
:class: tip
This topic accompanies these recorded lessons:

- Module 2, Lesson 1: *Python Basics Review*
- Module 2, Lesson 2: *Introduction to Pyomo*
- Module 2, Lesson 3: *Pyomo Model Components: Variables, Parameters and Sets*
- Module 2, Lesson 4: *Pyomo Model Components: Constraints and Objectives*
- Module 2, Lesson 5: *Solving Pyomo Models and Interpreting Results*
- Module 2, Lesson 6: *Solver Installation and Output Status*
```

:::{admonition} Learning Objectives
:class: tip

By the end of this topic, you should be able to:

- Translate a written formulation (sets, parameters, variables, objective, constraints) into a Pyomo `ConcreteModel`.
- Use indexed components and rules so that one model works for any number of stacks or constraints.
- Solve LP, MILP and NLP models with an appropriate solver (HiGHS, CBC, Ipopt) and read back variable values, objective values, duals and reduced costs.
- Check a solver's status and termination condition, and recognize an infeasible, unbounded or unfinished solve.
:::

In Topics 1.1 and 1.2 we wrote each optimization problem twice, first in mathematics and then as arrays for `scipy.optimize`. Writing the problem as arrays works, although it hides the model, because a constraint like "the loads add up to the demand" turns into a row of ones in a matrix that someone has to decode to understand what each column means. This is fine when there are only three variables, but it is not practical for a plant that has hundreds of units, several time periods and a dozen kinds of constraints.

An **algebraic modeling language** solves this problem by letting us write the model in terms that look like the mathematics, which means using sets of things, parameters that are indexed by those sets, and variables and equations that are defined over them. The language then builds the matrices and hands them to a solver. This separates two jobs that are easy to confuse, since *formulating* the problem is not the same as *solving* it, and once they are separated we can change solvers without touching the model or change the model without thinking about how the solver wants its input.

In this course we will use **Pyomo**, which is an open-source modeling language written in Python. Pyomo was started at Sandia National Laboratories by a team of mostly chemical engineers, and it is now maintained by developers at national laboratories and universities. Since Pyomo is ordinary Python, everything we know about Python still applies, and the data can come from any Python source, whether that is a dictionary, a spreadsheet read with pandas, or the output of another calculation.

Throughout this topic the running model will be the electrolyzer fleet of Topic 1.1, where three stacks A, B and C must together meet a hydrogen demand $D$ at least total power.

## Python for this course

The lectures open with a short review of Python for students coming from MATLAB, which we do not repeat here. If you need a refresher on variables, lists, loops, functions and NumPy arrays, work through DA4CHE Topic 1.1, *Python Basics* (https://www.medford.chbe.gatech.edu/DA4CHE/), since it covers the same ground and more.

The packages used in this book are NumPy (arrays and linear algebra), SciPy (scientific algorithms, including `scipy.optimize`), Matplotlib (plots), Pyomo (optimization modeling), and later scikit-learn and PyTorch (machine learning). These should all be installed in one environment, and the book's `environment.yml` lists the versions that we test against. Jupyter notebooks are the main way we work, because they keep code, results and explanation in one document, although for longer scripts a `.py` file edited in an IDE such as Spyder or VS Code and run from a console works equally well. Anaconda (or its lighter relative, Miniforge) manages packages and versions, and creating a separate environment for this course keeps its packages from interfering with others. Compared with MATLAB, which many engineers learn first, Python is free, open source and general purpose, with a large community sharing code, while the strengths of MATLAB are its integrated environment and its specialized toolboxes. The two differences that trip up MATLAB users most often are that Python indexes from 0, and that assigning a slice of a NumPy array creates a *view* of the same data rather than a copy (so use `.copy()` when you need a copy).

Before we get to Pyomo, it is worth looking at one Python data type that Pyomo leans on heavily, the **dictionary**. A list is indexed by position, but a dictionary is indexed by a *key*, which can be a string, a number, or a tuple. Optimization models are written over named things (stacks, species, customers, time periods), which makes dictionaries the natural way to store their data.

:::{admonition} Adapted material
:class: note
The dictionary and Pyomo-basics material in this topic is adapted from the course notebooks `Module 2/M2 - Introduction to Pyomo.ipynb`, `Module 2/M2-Examples.ipynb`, `pyomo solvers/Sets and Parameters.ipynb` and `pyomo solvers/Getting started with Pyomo and solvers.ipynb` (MIT License, © 2022 Data-Driven PSE Lab @ChBE-GT), rewritten for Pyomo 6.10 and the HiGHS and Ipopt solvers and moved onto the electrolyzer example.
:::

```{code-cell} ipython3
%matplotlib inline
import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('../settings/plot_style.mplstyle')
from _lib import electrolyzer as el

e_list = [1.20, 1.45, 1.70]                 # indexed by position: e_list[0] is stack A
e_dict = el.E_LINEAR                        # indexed by name
print(e_list[0], e_dict['A'])
print(e_dict)

# a two-index parameter is a dictionary keyed by tuples
loss = {(stack, kind): v for stack in el.NAMES
        for kind, v in (('ohmic r', el.R_OHMIC[stack]), ('parasitic a', el.A_PARASITIC[stack]))}
print(loss[('B', 'ohmic r')])
```

Asking for `e_dict[0]` would raise a `KeyError`, because the dictionary has no key `0`. This is actually helpful, since if the data are keyed by stack name there is no way for a model to silently confuse stack A with "the first column".

:::{exercise}
:label: ex-t13-dictionaries

Build a dictionary `full_load_mw` that maps each stack name to its full-load power in MW under the linear model, using `el.E_LINEAR` and `el.to_mw`. Then build a two-index dictionary keyed by `(stack, load)` for loads 0.25, 0.5 and 1.0 that stores the ohmic power $x + r_i x^2$ in per-unit.
:::

## A first Pyomo model

Every Pyomo model is built from the same five kinds of components, which correspond to the ingredients of a written formulation:

| Pyomo component | Role in the formulation |
|---|---|
| `Set` | the index sets: which stacks, which species, which time periods |
| `Param` | the data: efficiencies, capacities, demands |
| `Var` | the decision variables |
| `Objective` | the expression to minimize or maximize |
| `Constraint` | the equations and inequalities that must hold |

To complete the model we also need the **solver** that will be called, which a written formulation does not name.

Pyomo offers two ways to build a model. In an `AbstractModel` the structure is declared first and the data are supplied afterwards during the creation of an instance, whereas in a `ConcreteModel` each component is built at the moment of declaration, so the data have to exist beforehand. Concrete models are simpler, and this book uses concrete models throughout. When we want to use the same model for many data sets, we can put the concrete model inside a Python function that takes the data as arguments.

The first example in the lecture is a two-variable LP:

$$
\min_{x_1, x_2 \ge 0} \; x_1 + 2x_2 \quad \text{s.t.} \quad 3x_1 + 4x_2 \ge 1, \quad 2x_1 + 5x_2 \ge 2 .
$$

Solving this LP requires a solver, and solvers are separate programs that Pyomo talks to. In the next cell we therefore define a small helper that can find a solver whether it was installed with conda or with `idaes get-extensions`, and that can also cope with the case where no solver is installed. If a solver is missing the helper returns `None` rather than crashing, so that the rest of the page can still run with a graceful fallback. (The same helper is saved in the book's `_lib/solvers.py` for later topics.)

```{code-cell} ipython3
import os, shutil
import pyomo
import pyomo.environ as pyo

# where `idaes get-extensions` puts ipopt, cbc, bonmin, couenne; set IDAES_EXT if yours are elsewhere
IDAES_DIRS = [os.environ.get('IDAES_EXT', ''), os.path.expanduser('~/.idaes/bin'),
              os.path.join(os.environ.get('CONDA_PREFIX', ''), 'idaes-ext')]

def get_solver(name):
    """Return a Pyomo solver object, or None if it cannot be found."""
    if name.startswith('appsi_'):                       # solvers with a Python interface, e.g. appsi_highs
        solver = pyo.SolverFactory(name)
    else:                                               # solvers that run as an executable
        exe = shutil.which(name) or next((os.path.join(d, name) for d in IDAES_DIRS
                                          if d and os.path.isfile(os.path.join(d, name))), None)
        solver = pyo.SolverFactory(name, executable=exe) if exe else pyo.SolverFactory(name)
    return solver if solver.available(exception_flag=False) else None

highs = get_solver('appsi_highs')    # LP and MILP
ipopt = get_solver('ipopt')          # NLP
cbc = get_solver('cbc')              # LP and MILP (an alternative to HiGHS)
for name, s in (('HiGHS', highs), ('Ipopt', ipopt), ('CBC', cbc)):
    print(f"{name:6s}: {'found' if s is not None else 'NOT FOUND'}")
print('Pyomo', pyomo.__version__)
```

Now we can write the model itself. The import statement is `import pyomo.environ as pyo`, and every component then gets the prefix `pyo.`. In many online examples (including the lecture notebooks) the import is `from pyomo.environ import *` instead, which makes `ConcreteModel`, `Var` and so on available without a prefix. Both of these work, but with the prefixed form it is clear which names come from Pyomo, and there is no risk of clashing with names such as `sum` or `value`. As you read the model, compare each line with the corresponding part of the formulation.

```{code-cell} ipython3
model = pyo.ConcreteModel()

model.x1 = pyo.Var(within=pyo.NonNegativeReals)
model.x2 = pyo.Var(within=pyo.NonNegativeReals)

model.obj = pyo.Objective(expr=model.x1 + 2 * model.x2, sense=pyo.minimize)

model.con1 = pyo.Constraint(expr=3 * model.x1 + 4 * model.x2 >= 1)
model.con2 = pyo.Constraint(expr=2 * model.x1 + 5 * model.x2 >= 2)

results = highs.solve(model)
print(f"x1 = {pyo.value(model.x1):.3f}, x2 = {pyo.value(model.x2):.3f}, objective = {pyo.value(model.obj):.3f}")
```

The pieces of the model are:

- `pyo.ConcreteModel()` creates an empty model. We store it in a Python variable called `model`; the name is ours.
- `model.x1 = pyo.Var(...)` attaches a variable to the model. The attribute name (`x1`) becomes the component's name. `within=` gives its **domain**, the set of values it may take: `NonNegativeReals`, `Reals`, `Binary`, `NonNegativeIntegers`, and so on.
- `pyo.Objective(expr=..., sense=...)` declares what to optimize. `sense` is `pyo.minimize` (the default) or `pyo.maximize`.
- `pyo.Constraint(expr=...)` declares a constraint. Pyomo accepts `<=`, `>=` and `==`, but not strict `<` or `>`. A solver cannot represent a strict inequality, because there is no largest number below 1.
- `highs.solve(model)` sends the model to the solver and, by default, loads the solution back into the variables. `pyo.value(...)` reads a number out of a variable or an expression.

The optimum is $(x_1, x_2) = (0, 0.4)$ with objective 0.8. In the lecture the same model was solved with GLPK, an older open-source LP solver, while here we use HiGHS, which is a modern open-source LP and MILP solver that `scipy.optimize.linprog` and `milp` also use internally. The answer does not depend on which solver we choose, and this is exactly why we separate the model from the solver.

:::{exercise}
:label: ex-t13-first-model

Change the objective to $3x_1 + 2x_2$ and re-solve. Which constraint is now tight (holds with equality) at the optimum? Then call `model.pprint()` and find, in its output, the bounds and current value of each variable.
:::

## Sets, parameters and variables

Writing `model.x1` and `model.x2` by hand does not scale, since a fleet of 30 stacks should not need 30 lines. **Indexed components** solve this problem by letting us declare a set once, and then declare parameters, variables and constraints *over* the set.

### Sets

A `Set` is a collection of keys (numbers, strings or tuples) that is used to index other components. For example, `pyo.Set(initialize=['A', 'B', 'C'])` builds a set from a list, and `pyo.RangeSet(1, 10)` builds the integers 1 to 10. Sets can also be built from Python logic (such as a generator that keeps only some elements), the product of two sets, `model.I * model.J`, is the set of all pairs, and `len(model.S)` gives the number of elements. Sets combine with the usual set operators, which are union `|`, intersection `&`, difference `-` and the product `*`. There are two common mistakes that are worth knowing about. The first is that `pyo.Set([1, 2, 3])` does not build a set with those elements, because the list is read as an *index* for a family of sets, so you should write `pyo.Set(initialize=[1, 2, 3])` instead. The second is that a plain Python list assigned to the model (`model.S = [1, 2, 3]`) works for indexing but is not a Pyomo component, which means that Pyomo cannot check or print the list.

### Parameters

A `Param` holds data. An unindexed parameter acts like a number, while an indexed parameter acts like a dictionary and can be initialized directly from one. By default parameters are **immutable**, so once the model is built their values are baked into the expressions. If we declare `mutable=True`, the parameter is kept as a symbol whose value can be changed and re-solved without rebuilding the model, and since we plan to vary the demand $D$ this is the option we want for $D$. A `default=` value fills in any keys that are missing from the initializing dictionary.

### Variables

A `Var` represents a decision, and the most useful optional arguments are:

- `bounds=(lb, ub)`: simple lower and upper bounds. In Pyomo's bookkeeping, bounds are not constraints; the solver handles them separately and more efficiently.
- `domain=` (or `within=`): the kind of values allowed, such as `NonNegativeReals` or `Binary`.
- `initialize=`: a starting value. LP and MILP solvers ignore it. For nonlinear solvers it is the starting point, and it can decide which local minimum is found (Topic 1.1).

The value of a variable can be set by hand (`model.x['A'].value = 0.5`), the bounds of a variable can be read (`model.x['A'].ub`) and changed after construction (`model.x['A'].setub(0.8)`), and a variable can be **fixed** (`model.x['A'].fix(0)`) so that the solver treats it as a constant until it is unfixed. By fixing variables we can explore scenarios like a situation where stack A is offline, without having to rewrite the model.

Below are the sets, parameters and variables of the fleet LP from Topic 1.2, with power curve $P_i(x) = e_i x$:

```{code-cell} ipython3
m = pyo.ConcreteModel()
m.S = pyo.Set(initialize=el.NAMES)                     # the stacks
m.e = pyo.Param(m.S, initialize=el.E_LINEAR)           # specific energy, per-unit power per unit load
m.D = pyo.Param(initialize=1.2, mutable=True)          # demand, per-unit (1.2 = 240 kg/h)
m.x = pyo.Var(m.S, bounds=(0, 1), initialize=0.4)      # load of each stack, fraction of 200 kg/h

print(len(m.S), list(m.S))
print({i: m.e[i] for i in m.S}, 'D =', pyo.value(m.D))
m.x['A'].fix(0)
print('A fixed?', m.x['A'].fixed, ' value', m.x['A'].value)
m.x['A'].unfix()
```

In practice the model data usually live outside the model, in a spreadsheet, a database, or another Python module. The pandas package reads a spreadsheet into a `DataFrame` (`pd.read_excel('stacks.xlsx', index_col=0)`), which is a table with named rows and columns, and each column converts to exactly the dictionary that a `Param` wants. Here we build the table in code so that the page has no file dependency, and then use the table to declare the sets, parameters and variable bounds:

```{code-cell} ipython3
import pandas as pd

stacks = pd.DataFrame({'e': el.E_LINEAR, 'r': el.R_OHMIC, 'cap': {'A': 1.0, 'B': 1.0, 'C': 0.8}})
print(stacks)

md = pyo.ConcreteModel()
md.S = pyo.Set(initialize=stacks.index.tolist())
md.Hot = pyo.Set(initialize=['B', 'C'])                       # stacks that are already warm
md.e = pyo.Param(md.S, initialize=stacks['e'].to_dict())
md.x = pyo.Var(md.S, bounds=lambda m, i: (0, stacks.loc[i, 'cap']))
print('cold stacks:', list(md.S - md.Hot), '  warm and cheap:', [i for i in md.Hot if md.e[i] < 1.6])
print('upper bound of C:', md.x['C'].ub)
md.x['C'].setub(0.5)
print('after setub:     ', md.x['C'].ub)
```

:::{exercise}
:label: ex-t13-sets-params

Add a parameter `m.cap` indexed by `m.S` with capacities 1.0, 1.0 and 0.8, and redeclare `m.x` so that its upper bound comes from `m.cap`. (Hint: `bounds` can be a function `lambda m, i: (0, m.cap[i])`.) Then try to change `m.e['A']` after the model is built. What happens, and what would you change to allow it?
:::

## Constraints and objectives

### Constraints

A single constraint is declared with `expr=`, as in the first example, whereas an indexed constraint is declared over a set with a **rule**, which is a Python function that receives the model and an index and returns the expression for that index. Pyomo calls the rule once per element of the set, and if there should be no constraint at all for some index, the rule returns `pyo.Constraint.Skip`.

Pyomo also accepts **range constraints**, $\ell \le g(x) \le u$, which are written as `pyo.inequality(l, g, u)` or as a tuple `(l, g, u)`, while equality constraints use `==`.

### Objectives

An objective is declared in the same way, with `expr=` or `rule=`, plus `sense=`. Objectives are rarely indexed, and although a model may declare several objectives, only one of them may be active when the model is solved. For long expressions `pyo.quicksum` is a faster version of `sum`, and `pyo.sum_product(m.e, m.x)` builds $\sum_i e_i x_i$ in one call.

We can now finish the fleet model, which we wrap in a function because we will solve it several times with different data. The two **suffixes** at the end ask the solver to send back two extra pieces of information, namely the dual value of every constraint and the reduced cost of every variable. The meaning of these numbers is explained below, and we will rely on them throughout Topics 1.3–1.5.

```{code-cell} ipython3
def fleet_lp(D=1.2, e=el.E_LINEAR):
    """Linear-loss dispatch: min sum_i e_i x_i  s.t.  sum_i x_i = D,  0 <= x_i <= 1."""
    m = pyo.ConcreteModel()
    m.S = pyo.Set(initialize=list(e))
    m.e = pyo.Param(m.S, initialize=e)
    m.D = pyo.Param(initialize=D, mutable=True)
    m.x = pyo.Var(m.S, bounds=(0, 1))

    def demand_rule(m):
        return sum(m.x[i] for i in m.S) == m.D
    m.demand = pyo.Constraint(rule=demand_rule)

    m.power = pyo.Objective(expr=sum(m.e[i] * m.x[i] for i in m.S), sense=pyo.minimize)

    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)   # constraint duals
    m.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)     # variable reduced costs
    return m

m = fleet_lp()
m.demand.pprint()
m.power.pprint()
```

The printed expressions are the formulation that Pyomo generated from our rule, where the demand row says $x_A + x_B + x_C = D$ and the objective is $1.2x_A + 1.45x_B + 1.7x_C$.

:::{exercise}
:label: ex-t13-indexed-constraint

Add an indexed constraint `m.ramp` over `m.S` that limits each stack's load to at most 0.3 above a given current load `x_now = {'A': 0.7, 'B': 0.0, 'C': 0.5}`, and use `pyo.Constraint.Skip` so that stack B, which is cold, gets no ramp constraint. Print the constraint with `pprint()` to check that only two rows were built.
:::

## Solving models and interpreting results

There are two ways to solve a Pyomo model. Inside a script or notebook (which is how we work throughout), calling `solver.solve(model)` will build the input for the solver, run the solver, and load the results back into the model. If we pass `tee=True` the solver will also stream its own log to the screen, and this log is the first thing to look at when a solve misbehaves. From a terminal, the command `pyomo solve --solver=ipopt my_model.py` does the same for a file that defines a model, since it creates the model, applies the solver, loads the results and displays them. With the options `--stream-solver`, `--summary` and `--show-results` it will print the solver log, a summary of the model, and the full results, and within a script we can get similar information from `model.pprint()` and `model.display()`.

### An LP

Next we solve the fleet LP and read back three kinds of numbers, which are the loads, the total power, and the two suffixes.

```{code-cell} ipython3
m = fleet_lp(D=1.2)
res = highs.solve(m)
print('termination:', res.solver.termination_condition)
for i in m.S:
    print(f"  stack {i}: x = {pyo.value(m.x[i]):.3f} ({el.to_kg_h(pyo.value(m.x[i])):5.0f} kg/h), "
          f"reduced cost {m.rc[m.x[i]]:+.3f}")
P = pyo.value(m.power)
print(f"total power {P:.3f} per-unit = {el.to_mw(P):.2f} MW")
print(f"dual of the demand constraint: {m.dual[m.demand]:.3f}")

# what does the dual mean? re-solve with slightly more demand
m.D = 1.2 + 1e-3
highs.solve(m)
print(f"finite difference dP*/dD = {(pyo.value(m.power) - P) / 1e-3:.3f}")
```

The cheapest stack, A, runs at full load, while B covers the remaining 0.2 and C stays off, which gives a total power of 1.49 per-unit, or 9.74 MW.

The **dual** of a constraint is the rate at which the optimal objective changes when the right-hand side of the constraint changes. In this case the dual is 1.45, so one more unit of demand costs 1.45 more units of power, as the finite difference confirms. The **reduced cost** of a variable is the rate at which the objective would change if that variable were pushed away from its bound. The reduced cost of $+0.25$ for stack C says that forcing C to carry load would cost 0.25 more per unit than the current plan, and the value of $-0.25$ for stack A says that if A could go above full load, each extra unit would save 0.25.

These numbers carry the thermodynamics and the economics of the model, and we will read them carefully in Topic 1.4. For now the practical point is how to get them, which is to declare `Suffix(direction=IMPORT)` components named `dual` and `rc` before solving, and index them by the constraint or the variable. Not every solver interface returns every suffix, although HiGHS and Ipopt both return duals, and HiGHS also returns reduced costs.

Another way in which solvers and interfaces differ is the **sign** of the dual. For a minimization, Pyomo reports the derivative of the objective with respect to the right-hand side, so a demand that costs more to meet has a positive dual, whereas `scipy.optimize.linprog` follows a different convention for inequality rows (Topic 1.4). Since the finite difference is the reliable check, it is a good idea to perturb a right-hand side once and compare whenever you start using a new solver.

### An NLP

Next, we change one modeling choice, the loss model, to the ohmic curve $P_i(x) = x + r_i x^2$. This turns the model into a nonlinear program, so we switch to Ipopt (an interior-point solver for NLPs), and only the objective line and the solver need to change.

```{code-cell} ipython3
def fleet_ohmic(D=1.2):
    m = pyo.ConcreteModel()
    m.S = pyo.Set(initialize=el.NAMES)
    m.r = pyo.Param(m.S, initialize=el.R_OHMIC)
    m.D = pyo.Param(initialize=D, mutable=True)
    m.x = pyo.Var(m.S, bounds=(0, 1), initialize=D / 3)     # a starting point matters for NLPs
    m.demand = pyo.Constraint(expr=sum(m.x[i] for i in m.S) == m.D)
    m.power = pyo.Objective(expr=sum(m.x[i] + m.r[i] * m.x[i] ** 2 for i in m.S))
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return m

m2 = fleet_ohmic()
if ipopt is not None:
    res = ipopt.solve(m2)
    x_nlp = {i: pyo.value(m2.x[i]) for i in m2.S}
    lam = m2.dual[m2.demand]
    print('termination:', res.solver.termination_condition)
else:                                                      # fallback: the same model with scipy
    from scipy.optimize import minimize
    r = np.array([el.R_OHMIC[i] for i in el.NAMES])
    sol = minimize(lambda x: np.sum(x + r * x**2), np.full(3, 0.4), method='SLSQP', bounds=[(0, 1)] * 3,
                   constraints=[{'type': 'eq', 'fun': lambda x: x.sum() - 1.2}])
    x_nlp = dict(zip(el.NAMES, sol.x)); lam = 1 + 2 * r[0] * sol.x[0]
    print('Ipopt not found; solved with scipy SLSQP instead')
print({i: round(v, 4) for i, v in x_nlp.items()})
cost = sum(v + el.R_OHMIC[i] * v**2 for i, v in x_nlp.items())
print(f"total power {cost:.5f} per-unit = {el.to_mw(cost):.2f} MW;  demand dual {lam:.5f}")
print('marginal power 1 + 2 r_i x_i:', {i: round(1 + 2 * el.R_OHMIC[i] * v, 5) for i, v in x_nlp.items()})
```

With losses that grow with load, every stack now runs, at loads $(0.3818, 0.5455, 0.2727)$, and the total is 1.42909 per-unit (9.34 MW). This number should not be compared with the 1.49 from the LP, because the two loss models describe different stacks. Instead, what changed is the shape of the answer, since the optimum moved from a vertex (one stack full, one off) to an interior split. The demand dual is 1.38182, and the last line shows why this is the case, because at the optimum every running stack has the same marginal power $dP_i/dx_i = 1 + 2r_i x_i$, and that common value is the dual. This equal-marginal-cost rule is explained in Topic 2.1 and derived in Topic 2.2. The NLP example in the lectures is the Rosenbrock function, $(1 - x)^2 + 100(y - x^2)^2$, which is solved with Ipopt in the same few lines, and any smooth objective is written in the same way.

### A MILP

The integer example in the lecture is a knapsack problem with four tools that have values and weights, a weight limit of 14, and the goal of carrying the most value. This problem needs only binary variables and the `maximize` sense.

```{code-cell} ipython3
value = {'hammer': 8, 'wrench': 3, 'screwdriver': 6, 'towel': 11}
weight = {'hammer': 5, 'wrench': 7, 'screwdriver': 4, 'towel': 3}

k = pyo.ConcreteModel()
k.I = pyo.Set(initialize=list(value))
k.take = pyo.Var(k.I, within=pyo.Binary)
k.weight = pyo.Constraint(expr=sum(weight[i] * k.take[i] for i in k.I) <= 14)
k.value = pyo.Objective(expr=sum(value[i] * k.take[i] for i in k.I), sense=pyo.maximize)
highs.solve(k)
print('take:', [i for i in k.I if pyo.value(k.take[i]) > 0.5], ' value', pyo.value(k.value))
```

In the optimal solution the wrench is left behind. HiGHS handled the binaries without being told anything new, because it is the model (and not the solver call) that says which variables are integers. The MILP for the electrolyzer itself, with an on/off decision per stack, will have to wait until Topic 2.4.

:::{exercise}
:label: ex-t13-solve-loop

Using `fleet_lp`, build the model once and loop over demands $D = 0.2, 0.4, \ldots, 3.0$, changing only the mutable parameter `m.D` and re-solving. Plot the total power (MW) and the demand dual against the demand (kg/h). Where does the dual jump, and which stack is partly loaded on each flat stretch?
:::

## Solvers, status and termination

### Which solver for which problem

Any given solver handles some problem classes and not others, and the table in Topic 1.1 tells us which answers we can trust. The open-source solvers used in this book, along with a few commercial ones that are common in practice, are:

| Solver | Classes | Notes |
|---|---|---|
| HiGHS | LP, MILP (and convex QP) | open source; used through `appsi_highs`; also inside `scipy.optimize` |
| CBC, GLPK | LP, MILP | open source; older; CBC appears in Topic 2.5 |
| Ipopt | NLP (local) | open-source interior-point method; local optimum only, unless the problem is convex |
| Bonmin | MINLP (local) | open source; exact for convex MINLPs only |
| Couenne | MINLP (global) | open source; spatial branch-and-bound (Topic 2.6) |
| Gurobi, CPLEX | LP, MILP, some nonlinear | commercial; free academic licenses |
| BARON | MINLP (global) | commercial; the demo version is limited in size |

**Installation.** HiGHS comes with `pip install highspy` (or `conda install -c conda-forge highspy`), and Pyomo finds it automatically. Ipopt and CBC are executables, which can be installed with `conda install -c conda-forge ipopt coincbc`, or with `idaes get-extensions` (from the IDAES process-modeling package), which downloads Ipopt, CBC, Bonmin and Couenne together. Our `get_solver` helper checks both places. If a solver is missing, Pyomo will not raise an error when the model is built but only at solve time, so it is best to check whether the solver is available before solving, as the helper does.

### Status and termination condition

Every solve returns a results object, and before you read any variable you should read two of its fields:

- **Solver status** says whether the solver ran normally: `ok` (normal termination), `warning` (it finished, but something unusual happened), `error` (it stopped with an error), `aborted` (it was stopped from outside, by a time limit or an interrupt), or `unknown`.
- **Termination condition** says why it stopped. When things go well this is `optimal` (or `globallyOptimal` or `locallyOptimal`, for solvers that distinguish them). Other common values are `infeasible` (no point satisfies the constraints), `unbounded` (the objective can improve forever), `maxIterations` and `maxTimeLimit` (the solver ran out of its allowance before finishing), and `infeasibleOrUnbounded` (the solver knows one of the two holds but not which).

Pyomo defines a longer list of termination conditions, which can be grouped by their meaning:

| Group | Termination conditions |
|---|---|
| finished with a solution | `optimal`, `globallyOptimal`, `locallyOptimal`, `feasible` (a feasible point, optimality not proven), `intermediateNonInteger` |
| stopped by a limit | `maxIterations`, `maxTimeLimit`, `maxEvaluations` (e.g. branch-and-bound nodes), `minFunctionValue` (found a value below a target), `minStepLength` (steps became too small to continue) |
| proved something about the model | `infeasible`, `unbounded`, `infeasibleOrUnbounded`, `invalidProblem` (the model is the wrong type for the solver) |
| failed or inconclusive | `solverFailure`, `internalSolverError`, `error`, `licensingProblems`, `noSolution`, `other`, `unknown` |
| interrupted | `userInterrupt`, `resourceInterrupt` |

In code, you should compare against the enumerations rather than strings, by importing `from pyomo.opt import SolverStatus, TerminationCondition` and then testing `results.solver.termination_condition == TerminationCondition.optimal`. If you get a status of `error` or `unknown`, or a condition such as `internalSolverError`, that cannot be explained from the model, you may have found a solver bug, and the Pyomo and solver developers ask users to report these bugs.

When a solver stops for any reason other than optimality it may still leave numbers in the variables, but those numbers are meaningless. Pyomo provides `pyo.check_optimal_termination(results)` as a one-line test. To see the other cases, we will break the fleet model on purpose in three ways, passing `load_solutions=False` so that Pyomo does not try to load a solution that does not exist.

```{code-cell} ipython3
def report(label, res):
    ok = pyo.check_optimal_termination(res)
    print(f"{label:36s} status = {str(res.solver.status):8s} termination = {str(res.solver.termination_condition):14s}"
          f" usable: {ok}")

# 1. Infeasible: 700 kg/h of demand from three stacks of 200 kg/h each
report('demand D = 3.5 (above capacity)', highs.solve(fleet_lp(D=3.5), load_solutions=False))

# 2. Unbounded: sell hydrogen at 2.0 per unit with no capacity limit, and maximize profit
u = pyo.ConcreteModel()
u.S = pyo.Set(initialize=el.NAMES)
u.x = pyo.Var(u.S, within=pyo.NonNegativeReals)                     # the upper bound is forgotten
u.profit = pyo.Objective(expr=sum((2.0 - el.E_LINEAR[i]) * u.x[i] for i in u.S), sense=pyo.maximize)
report('profit, no capacity bound', highs.solve(u, load_solutions=False))

# 3. Out of iterations: the ohmic NLP with Ipopt allowed a single iteration
if ipopt is not None:
    report('ohmic NLP, max_iter = 1', ipopt.solve(fleet_ohmic(), options={'max_iter': 1}, load_solutions=False))
    report('ohmic NLP, default settings', ipopt.solve(fleet_ohmic()))
else:
    print('Ipopt not found; skipping the iteration-limit example')
```

Each type of failure tells us something different about the model. *Infeasible* means that the constraints contradict each other, and in this case the demand exceeds the capacity of the fleet, so the fix is in the data or the plant rather than in the solver. *Unbounded* almost always means that a constraint is missing, since every unit of hydrogen sold at 2.0 earns a margin, and without a capacity limit there is no ceiling on the profit. *Out of iterations* means that the solver stopped before finishing, and the honest response is to raise the limit, give a better starting point, or rescale the model, and then solve again. Only this last case is a solver problem, while the first two are modeling errors that have been diagnosed by the solver.

:::{exercise}
:label: ex-t13-status

Write a function `solve_or_explain(model, solver)` that solves a model and either returns a dictionary of variable values, when the termination is optimal, or prints a one-sentence diagnosis when it is not (infeasible: check the demand and capacities; unbounded: look for a missing bound; iteration or time limit: raise the limit or improve the start). Test it on the three broken models above. Then make the fleet LP infeasible in a different way, by fixing two stacks to zero with `fix(0)` at $D = 1.2$, and check that your function catches it.
:::

## Summary

- Pyomo writes an optimization model in the language of its formulation: sets, parameters, variables, an objective and constraints. The solver is a separate choice made at solve time.
- Indexed components and rules let one model serve any number of stacks. Mutable parameters and fixed variables let us ask "what if" without rebuilding.
- `Suffix(direction=IMPORT)` components named `dual` and `rc` bring back constraint duals and reduced costs. On the fleet LP the demand dual is 1.45 per-unit and stack C's reduced cost is 0.25; on the ohmic NLP the dual is 1.38182, the common marginal power of the running stacks. Check signs against a finite difference.
- Match the solver to the class: HiGHS for LP and MILP, Ipopt for NLP, Couenne for global MINLP.
- Read the status and termination condition before reading any variable. Infeasible and unbounded results are usually modeling errors; iteration limits are solver problems.

## Additional Reading

- M. L. Bynum, G. A. Hackebeil, W. E. Hart, C. D. Laird, B. L. Nicholson, J. D. Siirola, J.-P. Watson and D. L. Woodruff, *Pyomo — Optimization Modeling in Python*, 3rd ed., Springer (2021). The lectures cite the 2nd edition (Hart et al., 2017).
- The Pyomo documentation, https://pyomo.readthedocs.io, especially the sections on modeling components and on working with solver results.
- J. C. Kantor, *ND Pyomo Cookbook*, https://jckantor.github.io/ND-Pyomo-Cookbook/. Short worked Pyomo models from chemical engineering, including the production example used in the Module 3 lectures.
- Q. Huangfu and J. A. J. Hall, "Parallelizing the dual revised simplex method", *Mathematical Programming Computation* 10, 119–142 (2018). The simplex engine inside HiGHS; the HiGHS documentation is at https://highs.dev.
- A. Wächter and L. T. Biegler, "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming", *Mathematical Programming* 106, 25–57 (2006). The algorithm behind Ipopt.
