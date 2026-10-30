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

# Nonlinear and Mixed-Integer Optimization

Topics 1.1–1.5 dealt mostly with linear problems, where the answer sits at a vertex and any solver that finds a local optimum has found the global one. This module removes those comforts one at a time. Topics 2.1–2.3 treat nonlinear programs: convexity and the conditions for optimality, Lagrange multipliers and the KKT conditions (with the full derivation of the electrolysis equilibrium, where the multipliers turn out to be chemical potentials and the price of an electron), and duality. Topics 2.4–2.6 add discrete decisions and non-convexity: mixed-integer formulations, branch-and-bound, and global optimization with valid bounds.

By the end of the module the electrolyzer plant is a mixed-integer nonlinear program, which we solve with a certificate of global optimality and use to show why a local solver's "optimal" cannot be trusted on a non-convex problem.

Lecture weeks 4–5 (Modules 4–5).

## Topics

- {doc}`Topic2.1-Nonlinear_Programs_and_Convexity`
- {doc}`Topic2.2-Lagrange_Multipliers_and_KKT_Conditions`
- {doc}`Topic2.3-Duality`
- {doc}`Topic2.4-Mixed-Integer_Formulations`
- {doc}`Topic2.5-Branch_and_Bound`
- {doc}`Topic2.6-Non-convex_Problems_and_Bounding`
