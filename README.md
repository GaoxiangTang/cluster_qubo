# cluster\_qubo

# Hybrid QUBO solver with clustering‐based sub‐problem extraction

`cluster_qubo` is a research prototype that combines spectral clustering with a one‐layer Quantum Approximate Optimization Algorithm (QAOA) and classical local search to tackle large Quadratic Unconstrained Binary Optimization (QUBO) instances on near‑term quantum hardware.
The method is introduced in our preprint:

> Wending Zhao, Gaoxiang Tang  "Clustering‑based Sub‑QUBO Decomposition for Hybrid Quantum Optimization Algorithms", arXiv:2502.16212.

This repository makes the full source code and data available so that you can **reproduce every figure and table** in the paper and adapt the framework to your own problems.

---

## Features

* End‑to‑end pipeline for sub‑QUBO extraction, quantum evaluation, and classical post‑processing.
* Clean separation between **algorithm core** (`src/`) and **experiment scripts** (`experiments/`).
* Ready‑made Jupyter notebook (`demonstration/panels.ipynb`) that recreates all graphs in the arXiv paper.
* Modular design that lets you plug in alternative clustering methods, QAOA layers, or classical optimizers.
