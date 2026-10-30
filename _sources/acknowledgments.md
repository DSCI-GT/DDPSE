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

# Acknowledgments and Attribution

ChBE 4746/6746 was created and taught at Georgia Tech by Fani Boukouvala, whose recorded lectures this book accompanies and who is the book's first author. Its order follows her lectures. Parts 1 and 2, and the first half of Part 3, develop the optimization material around a running electrolyzer example designed by A.J. Medford. Topic 1.3 and the remaining topics adapt her lecture slides and the notebooks from the course repository [DDPSE/GT_DataDrivenPSE_Course](https://github.com/DDPSE/GT_DataDrivenPSE_Course), which is distributed under the MIT License:

> MIT License. Copyright (c) 2022 Data-Driven PSE Lab @ChBE-GT

Adapted code retains that notice. Each adapted section names its source notebook.

## Adapted course notebooks

The following notebooks from the course repository were adapted, with an attribution note in each adapted section. Every topic follows the coverage and order of the corresponding lecture slides; slide text and figures were not reproduced.

| Topic | Notebook (`GT_DataDrivenPSE_Course/`) | What was adapted |
|---|---|---|
| 1.3 | `Module 2/M2 - Introduction to Pyomo.ipynb`, `Module 2/M2-Examples.ipynb`, `pyomo solvers/` | First Pyomo model, sets and parameters, the knapsack MILP, solver installation and statuses (updated for Pyomo 6 and HiGHS) |
| 2.1 | `Module 4/Module 4 Example.ipynb` | The series-reaction residence-time problem |
| 3.1, 3.2 | `Module 6 - 8/Intro to Derivative-Free Optimization (Module6-7).ipynb` | Quadratic interpolation, positive spanning sets and bases, the simplex and Nelder–Mead |
| 3.5 | `Module 6 - 8/Optimization with Surrogates (Module 6-7-8).ipynb` | Latin hypercube sampling (now with `scipy.stats.qmc`), the scikit-learn surrogate-fitting pattern |
| 4.1 | `Module 9-10/Dealing with bigger and unstructured data.ipynb`, `Module 9-10/Outliers and Missing Values.ipynb` | Framing of unstructured data; deletion versus imputation; outlier types and Mahalanobis distance |
| 4.2 | `Module 9-10/Regression and Optimization with transformed reduced spaces (PCR, PLS).ipynb` and the cheddar-cheese data | Full-space versus principal-component regression and optimization in score space |
| 4.3, 4.4 | `Module 11 - 13/NeuralNets-HybridModels-PINNs.ipynb` | A trained network inside an optimization model (now PyTorch + OMLT); the PINN example (now PyTorch, reduced to one dimension) |
| 5.1, 5.2 | `Module 11 - 13/farmer_problem.ipynb` | The farmer's problem, stochastic and robust formulations (data from Birge and Louveaux) |

The cheddar-cheese data are the LaTrobe Valley cheddar study as distributed with Moore and McCabe, *Introduction to the Practice of Statistics* (also R `faraway::cheddar`).

Parts 1–3 are built around an electrolyzer example designed for this book. Its stylized power curve is inspired by, but does not reproduce, the alkaline electrolyzer models of Ulleberg (*Int. J. Hydrogen Energy* 28, 2003), Sánchez et al. (*Int. J. Hydrogen Energy* 43, 2018) and Baumhof, Raheli, Johnsen and Kazempour, "Optimization of Hybrid Power Plants: When Is a Detailed Electrolyzer Model Necessary?" (arXiv:2301.05310, 2023).

:::{admonition} Draft status
:class: warning
The attribution wording and the book's CC BY 4.0 license are pending confirmation by Prof. Boukouvala.
:::
