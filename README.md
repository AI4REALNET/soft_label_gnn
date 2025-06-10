# SoftGNN: Soft-Label Imitation Learning for Power Grid Topology Control

> **Code and models for the paper:**  
> *Learning Topology Actions for Power Grid Control: A Graph-Based Soft-Label Imitation Learning Approach*  
> Accepted at ECML PKDD 2025  
> Mohamed Hassouna, Clara Holzhüter, Malte Lehna, Matthijs de Jong, Jan Viebahn, Bernhard Sick, Christoph Scholz  
> [[Pre-print Paper PDF]](https://arxiv.org/abs/2503.15190)

---

## 🧠 Overview

This repository contains the official implementation of the **SoftGNN** agent, a novel **Graph Neural Network (GNN)**-based imitation learning framework for **power grid topology control**. The approach improves over traditional hard-label imitation learning by learning from **soft labels** that capture multiple viable actions for grid congestion mitigation. The agent operates in the **Grid2Op L2RPN WCCI 2022 environment**, outperforming both the expert and state-of-the-art RL agents.

---

## 🔍 Key Features

- **Soft-Label Generation**: Learn from a distribution over viable actions rather than a single expert action.
- **GNN-Based Architecture**: Use Graph Attention Networks (GAT) to encode power grid topology.
- **Action Feasibility Enhancements**: Improved substation reconfiguration support and line-disconnection handling.
- **N-1 Security Evaluation**: Post-hoc contingency-aware action selection for increased robustness.
- **Benchmarking**: Evaluation against greedy expert, and SOTA DRL agents in the L2RPN WCCI 2022 environment.

---

## 📬 Contact

For questions or collaborations, please contact:
mohamed.hassouna@iee.fraunhofer.de

