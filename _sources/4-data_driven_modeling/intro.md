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

# Data-Driven Modeling

The data-driven optimization topics assumed we could sample the black box wherever we liked. Real data is rarely so cooperative. This module is about building models for optimization from the data we actually have: data with gaps, noise and outliers (Topic 4.1), data in many correlated dimensions that are better described in a reduced space (Topic 4.2), and data combined with physical knowledge, either by correcting a mechanistic model with a learned component (Topic 4.3) or by building the physics into the learning itself (Topic 4.4).

The basic machine-learning tools are covered in the companion book, Data Analytics for Chemical Engineers. Here we focus on what changes when the model is going to be optimized.

Lecture weeks 9–12 (Modules 9–12).

## Topics

- {doc}`Topic4.1-Working_with_Real_Data`
- {doc}`Topic4.2-Optimization_in_Reduced_Spaces`
- {doc}`Topic4.3-Hybrid_Modeling`
- {doc}`Topic4.4-Physics-Informed_Machine_Learning`
