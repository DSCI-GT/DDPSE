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

# 5.1 Stochastic Programming

:::{admonition} Draft status: first draft
:class: warning
First draft (2026-10-30), written from the course design notes. Not yet reviewed.
:::

```{admonition} Lectures
:class: tip
This topic accompanies these recorded lessons:

- Module 13, Lesson 1: *Introduction to Uncertainty*
- Module 13, Lesson 2: *Stochastic Programming*
- Module 13, Lesson 3: *Stochastic Programming Example*
```

:::{admonition} Learning Objectives
:class: tip

By the end of this topic, you should be able to:

- Identify uncertain parameters in an optimization model and choose a description for them (distribution, scenarios or bounds).
- Separate the decisions in a problem into first-stage decisions, made before the uncertainty is revealed, and second-stage (recourse) decisions, made after.
- Write the deterministic equivalent of a two-stage stochastic linear program and solve it in Pyomo.
- Compute the expected value of perfect information and the value of the stochastic solution, and explain what each measures.
:::

Every model in this book so far has treated its parameters as known: the stack resistances, the demand, the rate constants. Real parameters are measured with error or describe the future, and an optimum computed for one set of values can perform badly, or not be feasible at all, when the values turn out different. This bonus part asks how to optimize when we know that we do not know. This topic covers stochastic programming, which optimizes the expected outcome over a set of possible futures; {doc}`Topic5.2-Robust_Optimization` covers robust optimization, which protects against the worst one.

## Uncertainty in optimization

An **uncertain parameter** is a parameter of an optimization model whose value is not known when the decision is made, either because it is measured with error or because it describes something that has not happened yet. Examples are the price of a raw material over the next ten months, a rate constant fitted to experiments, or the sulfur content of the next crude-oil cargo. The lectures point out that this is a different use of the word from the data-driven optimization topics, where "uncertainty" meant that we lacked the equations of the model. Here the equations are known and the numbers in them are not.

The model of an uncertain parameter depends on how much information is available:

- **A continuous probability distribution**, such as a normal distribution with a given mean and variance, when we know a lot.
- **A discrete distribution over scenarios**, for example three possible yields with probabilities 20%, 50% and 30%. A scenario is one complete realization of all the uncertain parameters.
- **Bounds or a set**, such as "the yield is within 20% of its mean", when we know only the range of possible values and nothing about how likely each is.
- Other descriptions exist, such as **fuzzy sets**, which grade how plausible each value is without a probability.

For scenarios $\xi_1, \ldots, \xi_S$ with probabilities $p_s$ (summing to one), the probability distribution is a set of spikes at the scenario values, and the expected value of any quantity is the weighted sum $\mathbb{E}[\xi] = \sum_s p_s\, \xi_s$. This weighted sum is the only piece of probability that will be needed for the rest of the topic.

The description that is chosen decides the formulation. Probabilities lead to **stochastic programming**, which optimizes the expected value of the objective over the scenarios and tends to give solutions that do well on average, so it is suited to planning problems that are repeated many times. Sets lead to **robust optimization**, which optimizes against the worst value in the set and gives conservative solutions, so it is suited to problems where feasibility is critical. Other frameworks, such as chance-constrained programming (where constraints must hold with a given probability) and flexibility analysis, are mentioned in the lectures but are beyond the scope of this book. Since uncertainty is everywhere, optimization under uncertainty is applied everywhere; electric-power generation, inventory management, portfolio selection, facility planning, capacity expansion and the analysis of biological systems are a few of the areas that are listed in the lectures, and stochastic programming in particular is standard in financial planning, capacity expansion and airline planning.

A note on terminology is needed here. The "stochastic methods" of {doc}`../3-data_driven_optimization/Topic3.3-Global_Sampling_and_Evolutionary_Methods` (simulated annealing, genetic algorithms, particle swarms) are *randomized search algorithms*, where the randomness is in the method and the problem itself is deterministic. In **stochastic programming** the situation is reversed: the randomness is in the parameters of the problem, and the method that is used to solve the deterministic equivalent (here, an LP or MILP solver) is deterministic. The two ideas can meet, as when a noisy black box (Topic 5.2) is searched by a randomized method, but the terms refer to different things.

:::{exercise}
:label: ex-t51-describe

For the electrolyzer fleet of Topics 1.1–3.6, list three parameters that are uncertain in practice. For each, say whether you would describe it by a distribution, by scenarios or by bounds, and whether it is uncertain because of measurement error or because it lies in the future.
:::

## Two-stage stochastic programs

The key modeling step is to ask *when* each decision is made relative to *when* the uncertainty is resolved. In the example from the lectures, a process makes C from A and B; A and B must be bought on day 1, extra B can be bought on day 5, and the price of B on day 5 is not known until day 5.

- **First-stage** decisions, also called **here-and-now** decisions, are made before the uncertainty is revealed (the day-1 purchases). They cannot depend on the outcome, so they have to work reasonably well in every scenario.
- **Second-stage** decisions, also called **recourse** decisions, are made after the uncertainty is revealed (the day-5 purchase). They can be different in each scenario, and they are how the plan adapts to what actually happened.

A problem that has only first-stage decisions is a single-stage stochastic program, while a problem with both kinds of decisions is a **two-stage** program, which is the most studied class. If the uncertainty is revealed in several steps with decisions in between, the problem is multi-stage.

A two-stage stochastic linear program, with first-stage variables $x$, second-stage variables $y$ and uncertain parameters $\xi$, can be written as

$$
\min_{x \ge 0}\; c^\top x + \mathbb{E}_\xi\big[Q(x, \xi)\big] \quad \text{s.t.} \quad A x = b, \qquad
Q(x, \xi) = \min_{y \ge 0} \big\{ q(\xi)^\top y \;:\; W y = h(\xi) - T(\xi)\, x \big\}.
$$

The function $Q$ is the cost of the best recourse once $\xi$ is known and $x$ is fixed. When $\xi$ takes a finite number of scenario values $\xi_s$ with probabilities $p_s$, the expectation becomes a weighted sum, and the problem can be written as one ordinary LP, called the **deterministic equivalent**, which has one copy of the second-stage variables and constraints for each scenario:

$$
\begin{aligned}
\min_{x,\, y_1, \ldots, y_S \ge 0} \quad & c^\top x + \sum_{s=1}^{S} p_s\, q_s^\top y_s \\
\text{s.t.} \quad & A x = b, \\
& T_s x + W y_s = h_s, \qquad s = 1, \ldots, S.
\end{aligned}
$$

The first-stage $x$ is shared by every scenario, which is what forces it to be a single decision, while the $y_s$ are free to differ. The price of this formulation is size: the number of second-stage variables and constraints grows linearly with the number of scenarios, and the number of scenarios grows exponentially with the number of independent uncertain parameters. Large instances are solved by decomposition methods that exploit the block structure (Benders decomposition, which is also called the L-shaped method, and progressive hedging), but these methods are beyond the scope of this book.

:::{exercise}
:label: ex-t51-size

A deterministic model has 50 first-stage variables, 200 second-stage variables and 150 constraints that involve second-stage variables. Each of 6 uncertain parameters takes one of 3 values independently. How many scenarios are there, and how many variables and constraints does the deterministic equivalent have?
:::

## The farmer's problem

:::{admonition} Adapted from the course notebook
:class: note
Adapted from `farmer_problem.ipynb` (MIT License, © 2022 Data-Driven PSE Lab @ChBE-GT), which follows Section 1.1 of Birge and Louveaux, *Introduction to Stochastic Programming*. Rewritten as one scenario-indexed Pyomo model, with the two sugar-beet price tiers as separate sales variables, solved with HiGHS instead of GLPK.
:::

The lectures use the classic example of Birge and Louveaux, where a farmer has 500 acres that must be divided among wheat, corn and sugar beets before the growing season.

- Planting costs \$150, \$230 and \$260 per acre for wheat, corn and beets.
- The cattle need at least 200 t of wheat and 240 t of corn. Any shortfall must be bought from a wholesaler at \$238/t (wheat) and \$210/t (corn).
- Surplus wheat and corn sell at \$170/t and \$150/t. Beets sell at \$36/t up to a quota of 6000 t, and at \$10/t beyond it.
- The mean yields are 2.5, 3 and 20 t/acre. Weather makes them uncertain: with equal probability, all yields are 20% below the mean, at the mean, or 20% above it.

The acreage is the first-stage decision, since it is fixed at planting, while sales and purchases are second-stage decisions that are made after the harvest reveals the yields. A single function can be written that builds the deterministic equivalent for any list of scenarios and is also able to fix the acreage, which is needed in order to evaluate a given plan.

```{code-cell} ipython3
%matplotlib inline
import sys
sys.path.insert(0, '..')
import numpy as np
import pyomo.environ as pyo
from _lib.part4b_tools import solve_linear

CROPS = ['wheat', 'corn', 'beets']
FEED = ['wheat', 'corn']
YIELD = {'wheat': 2.5, 'corn': 3.0, 'beets': 20.0}       # t/acre, mean
PLANT = {'wheat': 150, 'corn': 230, 'beets': 260}       # $/acre
SELL = {'wheat': 170, 'corn': 150}                      # $/t
BUY = {'wheat': 238, 'corn': 210}                       # $/t
NEED = {'wheat': 200, 'corn': 240}                      # t
LAND, QUOTA, BEET_HI, BEET_LO = 500, 6000, 36, 10

def farmer(scenarios, acres=None):
    """Deterministic equivalent. scenarios: {name: (probability, yield factor)}.
    If acres is given, the first stage is fixed and only the recourse is optimized."""
    S = list(scenarios)
    m = pyo.ConcreteModel()
    m.x = pyo.Var(CROPS, within=pyo.NonNegativeReals)                      # first stage: acres
    m.sell = pyo.Var(FEED, S, within=pyo.NonNegativeReals)                 # second stage: t
    m.buy = pyo.Var(FEED, S, within=pyo.NonNegativeReals)
    m.beet_hi = pyo.Var(S, bounds=(0, QUOTA))
    m.beet_lo = pyo.Var(S, within=pyo.NonNegativeReals)
    m.land = pyo.Constraint(expr=sum(m.x[c] for c in CROPS) <= LAND)
    m.feed = pyo.Constraint(FEED, S, rule=lambda m, c, s:
                            scenarios[s][1] * YIELD[c] * m.x[c] + m.buy[c, s] - m.sell[c, s] >= NEED[c])
    m.beets = pyo.Constraint(S, rule=lambda m, s:
                             m.beet_hi[s] + m.beet_lo[s] <= scenarios[s][1] * YIELD['beets'] * m.x['beets'])
    if acres is not None:
        for c in CROPS:
            m.x[c].fix(acres[c])
    revenue = {s: sum(SELL[c] * m.sell[c, s] - BUY[c] * m.buy[c, s] for c in FEED)
               + BEET_HI * m.beet_hi[s] + BEET_LO * m.beet_lo[s] for s in S}
    m.profit = pyo.Objective(sense=pyo.maximize, expr=sum(scenarios[s][0] * revenue[s] for s in S)
                             - sum(PLANT[c] * m.x[c] for c in CROPS))
    profit = solve_linear(m)
    return profit, {c: pyo.value(m.x[c]) for c in CROPS}

SCEN = {'20% low': (1 / 3, 0.8), 'at the mean': (1 / 3, 1.0), '20% high': (1 / 3, 1.2)}

z_mean, x_mean = farmer({'mean': (1.0, 1.0)})
print("deterministic plan (mean yields):", {c: round(v) for c, v in x_mean.items()}, f"acres, profit ${z_mean:,.0f}")
z_eval = {s: farmer({s: (1.0, f)}, acres=x_mean)[0] for s, (p, f) in SCEN.items()}
for s, z in z_eval.items():
    print(f"  the same plan if yields are {s:11s}: profit ${z:,.0f}")
EEV = sum(SCEN[s][0] * z for s, z in z_eval.items())
print(f"expected profit of the deterministic plan: ${EEV:,.0f}")
```

With mean yields the best plan is 120 acres of wheat, 80 of corn and 300 of beets, for a profit of \$118,600. The beet acreage is exactly what is needed to fill the 6000 t quota at 20 t/acre, and the corn acreage exactly meets the 240 t needed by the cattle. This is typical of an LP optimum, which sits on a vertex and leaves no slack anywhere; with no slack, there is no margin for a bad year. If the yields come in 20% low the farm is short of corn and has to buy it at a premium, so the profit falls to \$55,120, and if the yields come in high the extra beets beyond the quota sell for only \$10/t. Over the three equally likely scenarios the expected profit of the plan is \$107,240, or \$11,360 less than the deterministic model promised.

That gap is the first lesson of the lectures: the deterministic optimum overstates what the plan will earn, because it was optimized for a future that is only one of the possibilities.

The next step is to let the model see all three scenarios at once, where the acreage is shared and the sales and purchases are given one copy per scenario.

```{code-cell} ipython3
z_sp, x_sp = farmer(SCEN)
print("stochastic plan:", {c: round(v) for c, v in x_sp.items()}, f"acres, expected profit ${z_sp:,.0f}")
for s, (p, f) in SCEN.items():
    print(f"  stochastic plan if yields are {s:11s}: profit ${farmer({s: (1.0, f)}, acres=x_sp)[0]:,.0f}")
```

The stochastic plan plants 170 acres of wheat, 80 of corn and 250 of beets, which moves 50 acres from beets to wheat. Fewer beets means that less is wasted at \$10/t in a good year, and more wheat is a hedge, since surplus wheat can always be sold at \$170/t. The expected profit is \$108,390. The stochastic plan earns less than the deterministic plan in a mean year (\$109,350 against \$118,600) and in a bad year (\$48,820 against \$55,120), but much more in a good year (\$167,000 against \$148,000), when the beets of the deterministic plan overflow the quota. In other words, the stochastic plan gives up a little in two scenarios in order to gain a lot in the third, and comes out ahead on average.

:::{exercise}
:label: ex-t51-probabilities

Change the scenario probabilities to 0.5 (20% low), 0.3 (at the mean) and 0.2 (20% high), as a pessimistic farmer might. Re-solve the stochastic program. Which acreages change, and in which direction? Explain the change in terms of what each crop's recourse costs in a bad year.
:::

## The value of information and of the stochastic solution

Two numbers are used to summarize how much the uncertainty matters, and both of them compare the stochastic solution with an alternative. For a problem where profit is maximized:

- The **recourse problem** value, RP, is the expected profit of the stochastic solution: \$108,390 here.
- The **expected value of the expected-value solution**, EEV, is the expected profit of the plan computed for mean yields: \$107,240.
- The **wait-and-see** value, WS, is the expected profit we would earn if we could learn each year's yields before planting, and so plant the best plan for each scenario. It is the average of three deterministic optima.

From these, the **value of the stochastic solution** is $\text{VSS} = \text{RP} - \text{EEV}$, which is the gain from modeling the uncertainty instead of planning for the mean. The **expected value of perfect information** is $\text{EVPI} = \text{WS} - \text{RP}$, which is the worth of a perfect forecast, and therefore the most it would make sense to pay for better information. For a maximization problem $\text{WS} \ge \text{RP} \ge \text{EEV}$, so both of these quantities are nonnegative.

```{code-cell} ipython3
ws = {s: farmer({s: (1.0, f)}) for s, (p, f) in SCEN.items()}
for s, (z, x) in ws.items():
    print(f"perfect forecast, yields {s:11s}: plant {({c: round(v) for c, v in x.items()})}, profit ${z:,.0f}")
WS = sum(SCEN[s][0] * ws[s][0] for s in SCEN)
print(f"\nWS  = ${WS:,.0f}\nRP  = ${z_sp:,.0f}\nEEV = ${EEV:,.0f}")
print(f"EVPI = WS - RP  = ${WS - z_sp:,.0f}")
print(f"VSS  = RP - EEV = ${z_sp - EEV:,.0f}")
```

A perfect weather forecast would be worth about \$7,016 a year to this farmer, and modeling the uncertainty, without any better forecast, recovers \$1,150 of this value. The VSS is the payoff for the modeling effort in this topic, while the EVPI is the ceiling on the value added by any amount of forecasting. Both are small fractions of the profit here, but the gap between the deterministic promise and the expected outcome (\$11,360) is not small, and this gap is hidden by a deterministic model.

The wait-and-see plans show why no single plan can match WS. The beet acreage that exactly fills the quota is $6000/(20 f)$ for a yield factor $f$, which is 375 acres in a bad year, 300 in a mean year and 250 in a good one. In a bad year the best plan therefore plants 375 acres of beets and only 25 of corn and buys most of the corn for the cattle, while in a good year it plants 250 acres of beets and puts the rest into wheat. A first-stage decision has to compromise between these plans, and the 250 acres of beets in the stochastic plan show which way it leans, toward the good-year acreage. An acre of beets beyond the quota loses money (it costs \$260 to plant, and the 16 to 24 t that it yields sell at \$10/t), whereas an acre that is moved to wheat earns its keep in every scenario.

:::{exercise}
:label: ex-t51-evpi

Replace the three scenarios with yields drawn from a continuous distribution: a yield factor uniform on $[0.8, 1.2]$, the same for all crops. Draw $N$ samples with a fixed seed, give each probability $1/N$, and solve the stochastic program (this is the sample average approximation). Plot the expected profit and the wheat acreage against $N$ for $N = 3, 10, 30, 100$. Do they settle down, and to values near the three-scenario answer?
:::

## The electrolyzer fleet under uncertain demand

The same structure fits the running example. Suppose that the plant must decide an hour ahead which of the three stacks to commit (warm up and bring on line), because a stack cannot start instantly, and that the hydrogen demand for that hour is uncertain. Committing a stack costs the on-cost $F_\mathrm{ON}$ of that stack whether or not it is then used. Once the demand is known, the committed stacks are dispatched with the linear power curves of Topic 1.4, and any hydrogen that the committed stacks cannot make is bought from outside.

- First stage: the commitments $u_i \in \{0, 1\}$, $i \in \{A, B, C\}$.
- Uncertainty: demand $D$ is 0.8, 1.2 or 2.4 (160, 240 or 480 kg/h) with probabilities 0.3, 0.4 and 0.3. The mean is 1.44.
- Second stage: loads $0 \le x_{is} \le u_i$ and a purchase $b_s \ge 0$ with $\sum_i x_{is} + b_s \ge D_s$.

The cost is priced in per-unit power (1 per-unit = 6.53 MW, as in Topic 1.1), and the price of bought hydrogen is set at 3.0 per-unit, above the 1.70 of the most expensive stack. This is an illustrative number that is chosen so that buying is expensive enough to matter.

```{code-cell} ipython3
from _lib import electrolyzer as el

C_BUY = 3.0                                                    # per-unit power per unit of H2 bought
DEMAND = {'low': (0.3, 0.8), 'mid': (0.4, 1.2), 'high': (0.3, 2.4)}

def fleet(scenarios, commit=None):
    """Two-stage unit commitment: commit stacks now, dispatch and buy after demand is known."""
    S = list(scenarios)
    m = pyo.ConcreteModel()
    m.u = pyo.Var(el.NAMES, within=pyo.Binary)
    m.x = pyo.Var(el.NAMES, S, bounds=(0, 1))
    m.buy = pyo.Var(S, within=pyo.NonNegativeReals)
    m.cap = pyo.Constraint(el.NAMES, S, rule=lambda m, i, s: m.x[i, s] <= m.u[i])
    m.dem = pyo.Constraint(S, rule=lambda m, s: sum(m.x[i, s] for i in el.NAMES) + m.buy[s] >= scenarios[s][1])
    if commit is not None:
        for i in el.NAMES:
            m.u[i].fix(commit[i])
    m.cost = pyo.Objective(expr=sum(el.F_ON * m.u[i] for i in el.NAMES) + sum(
        scenarios[s][0] * (sum(el.E_LINEAR[i] * m.x[i, s] for i in el.NAMES) + C_BUY * m.buy[s]) for s in S))
    cost = solve_linear(m)
    return cost, {i: round(pyo.value(m.u[i])) for i in el.NAMES}

D_mean = sum(p * d for p, d in DEMAND.values())
_, u_mean = fleet({'mean': (1.0, D_mean)})
EEV_f = sum(p * fleet({s: (1.0, d)}, commit=u_mean)[0] for s, (p, d) in DEMAND.items())
RP_f, u_sp = fleet(DEMAND)
WS_f = sum(p * fleet({s: (1.0, d)})[0] for s, (p, d) in DEMAND.items())
print(f"mean demand {D_mean:.2f} ({el.to_kg_h(D_mean):.0f} kg/h): commit {u_mean}, expected cost {EEV_f:.3f}")
print(f"stochastic program:          commit {u_sp}, expected cost {RP_f:.3f}")
print(f"wait-and-see expected cost {WS_f:.3f}")
print(f"VSS  = EEV - RP = {EEV_f - RP_f:.3f} per-unit = {el.to_mw(EEV_f - RP_f):.2f} MW")
print(f"EVPI = RP - WS  = {RP_f - WS_f:.3f} per-unit = {el.to_mw(RP_f - WS_f):.2f} MW")
```

Planning for the mean demand of 1.44 commits stacks A and B, which together can make up to 2.0, whereas the stochastic program also commits stack C. In the high scenario, which has a probability of 0.3, the demand exceeds what A and B can make by 0.4, and covering that shortfall with C instead of buying saves $3.0 - 1.70 = 1.30$ per unit, or an expected $0.3 \times 0.4 \times 1.30 = 0.156$, which is more than the on-cost of 0.12 for C. The difference, 0.036 per-unit, is the VSS. For minimization the inequalities are reversed, $\text{WS} \le \text{RP} \le \text{EEV}$, so we write $\text{VSS} = \text{EEV} - \text{RP}$ and $\text{EVPI} = \text{RP} - \text{WS}$.

In energy terms, the VSS is an expected saving of about 0.24 MW of average power, and a perfect demand forecast would be worth about 0.78 MW.

In economic terms, the stochastic plan pays a certain cost (the on-cost of C, every hour) in order to avoid an uncertain cost that is larger (buying hydrogen in the high-demand hours). This is a form of insurance, which the stochastic program is able to price correctly because the probabilities are included in the model, while the deterministic plan is unable to price it because only the mean is included.

:::{exercise}
:label: ex-t51-fleet

Find the price of bought hydrogen, `C_BUY`, below which the stochastic program stops committing stack C, first by reasoning from the numbers above and then by solving `fleet` for a range of prices. What are the VSS and EVPI at that price?
:::

## Summary

- An uncertain parameter is one whose value is unknown when the decision is made. It can be described by a distribution, by scenarios, or by a set; the description decides the formulation.
- Two-stage stochastic programs separate first-stage (here-and-now) decisions from second-stage (recourse) decisions. With discrete scenarios, the deterministic equivalent is one LP or MILP with a copy of the recourse for each scenario.
- On the farmer's problem, the mean-yield plan promises \$118,600 but earns \$107,240 on average. The stochastic plan earns \$108,390: a VSS of \$1,150. A perfect forecast would add \$7,016 (the EVPI).
- On the electrolyzer fleet, the stochastic plan commits a third stack as insurance against high demand, with a VSS of 0.036 per-unit (0.24 MW).
- Stochastic programming needs probabilities. When we know only the range of the uncertain parameters, or when a constraint must never be violated, we turn to {doc}`Topic5.2-Robust_Optimization`.

## Additional Reading

- J. R. Birge and F. Louveaux, *Introduction to Stochastic Programming*, 2nd ed., Springer (2011). Section 1.1 is the farmer's problem.
- A. Shapiro, D. Dentcheva and A. Ruszczyński, *Lectures on Stochastic Programming: Modeling and Theory*, SIAM (2009).
- N. V. Sahinidis, "Optimization under uncertainty: state-of-the-art and opportunities," *Computers & Chemical Engineering* 28 (2004) 971–983.
- I. E. Grossmann, R. M. Apap, B. A. Calfa, P. García-Herreros and Q. Zhang, "Recent advances in mathematical programming techniques for the optimization of process systems under uncertainty," *Computers & Chemical Engineering* 91 (2016) 3–14.
- W. E. Hart, C. D. Laird, J.-P. Watson, D. L. Woodruff, G. A. Hackebeil, B. L. Nicholson and J. D. Siirola, *Pyomo — Optimization Modeling in Python*, 2nd ed., Springer (2017).
