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
---

# Summary

NeuroSTF is a novel deep learning framework designed for multivariate time-series forecasting. It integrates a neuroplastic sparse attention mechanism inspired by principles of brain plasticity, dynamically pruning and regrowing attention weights during training. This adaptive sparse attention reduces computational complexity while preserving long-range dependency modeling. Combined with temporal convolution and MLP layers, NeuroSTF efficiently captures complex temporal patterns and improves generalization in forecasting tasks.

# Statement of need

Multivariate time-series forecasting is a critical component in domains such as energy management, finance, and environmental monitoring. Traditional transformer models achieve high accuracy but often incur substantial computational and memory costs, limiting their deployment in resource-constrained environments or for very long sequences. NeuroSTF addresses these challenges by leveraging neuroplasticity-inspired sparse attention mechanisms that dynamically adapt during training, significantly improving efficiency and model generalization. This biologically motivated approach enables practical, scalable forecasting solutions where both accuracy and efficiency are essential.

# Installation

NeuroSTF requires Python 3.7 or higher. Install dependencies using:

```bash
pip install -r requirements.txt

python neurostf.py path/to/ETTh1.csv

@article{pal2025neurostf,
  title={NeuroSTF: Neuroplastic Sparse Attention for Time-Series Forecasting},
  author={Ovi Pal},
  journal={Journal of Open Source Software},
  year={2025},
  volume={},
  number={},
  pages={},
  doi={}
}

