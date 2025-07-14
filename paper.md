---
title: "NeuroSTF: Neuroplastic Sparse Attention for Time-Series Forecasting"
authors:
  - name: Ovi Pal
    orcid: 0009-0002-6849-162X
    affiliation: Independent Researcher
date: 2025-07-14
tags:
  - time-series forecasting
  - deep learning
  - sparse attention
  - neuroplasticity
  - multivariate forecasting
bibliography: paper.bib
---

# Summary

NeuroSTF is a deep learning framework for multivariate time-series forecasting that leverages neuroplastic sparse attention mechanisms inspired by brain plasticity. By dynamically pruning and regrowing attention weights during training, the model efficiently captures long-range dependencies with improved generalization and computational efficiency. The architecture also incorporates temporal convolution and MLP layers to enhance temporal feature extraction.

# Statement of need

Time-series forecasting is critical in numerous domains including energy, finance, weather, and healthcare. Existing transformer-based models are powerful but often computationally expensive and lack efficient mechanisms to adaptively focus on relevant temporal patterns. NeuroSTF addresses these challenges by introducing a biologically inspired sparse attention mechanism that dynamically evolves during training, reducing computational overhead while maintaining or improving accuracy. This makes NeuroSTF particularly suited for scenarios with resource constraints and long sequence forecasting tasks.

# Installation

NeuroSTF requires Python 3.7+ and several common scientific computing libraries. Install dependencies with:

```bash
pip install -r requirements.txt
