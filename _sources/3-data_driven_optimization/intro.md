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

# Data-Driven Optimization

Until now the optimizer could see the equations. In this module it cannot: the objective is a black box that we can only evaluate, as when it is a simulation or an experiment. Topics 3.1–3.3 cover methods that work from samples alone, from local direct search to global sampling and evolutionary methods, and show how a Lipschitz constant lets a sampling method prove its answer the way branch-and-bound did. Topic 3.4 treats constraints, which turn out to decide what a black-box method can find at all. Topics 3.5 and 3.6 build surrogate models of the black box and optimize them, ending with a comparison of finding an optimum versus proving it.

The electrolyzer plant from the first two modules is the black box throughout, so the methods can be compared against answers we already know.

Lecture weeks 6–8 (Modules 6–8).

## Topics

- {doc}`Topic3.1-Optimization_without_Equations`
- {doc}`Topic3.2-Direct-Search_Methods`
- {doc}`Topic3.3-Global_Sampling_and_Evolutionary_Methods`
- {doc}`Topic3.4-Constraints_in_Data-Driven_Optimization`
- {doc}`Topic3.5-Surrogate_Models`
- {doc}`Topic3.6-Surrogate-Based_Optimization`
