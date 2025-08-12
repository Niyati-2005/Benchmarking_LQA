# LQA Benchmarking for NP-Complete Problems

This repository benchmarks the **Local Quantum Annealing (LQA)** algorithm, as introduced in  
**Bowles et al.** ([arXiv:2108.08064v2](https://arxiv.org/abs/2108.08064)),  
on a set of NP-complete problems for comparative analysis against results from the *Turing.co* paper and other state-of-the-art solvers.

> **Attribution Notice**  
> The LQA algorithm implementation in this repository is **adapted directly from Bowles et al.**  
> All credit for the algorithm design and theoretical framework belongs to the original authors.  
> This implementation is included **solely for benchmarking and reproducibility**.

---

## 📌 Benchmarked Problems
- **Max-Cut**
- **NAE-3-SAT (Not-All-Equal 3-Satisfiability)**
- **Sherrington-Kirkpatrick (SK) Spin Glass Model**

---

## 🎯 Purpose
The objective of this project is to:
1. Evaluate Bowles’ LQA on classical NP-complete problems.
2. Compare its performance to other solvers, including:
   - Gurobi
   - Simulated Annealing
   - Simulated Bifurcation
   - Other quantum-inspired algorithms
3. Provide reproducible benchmarking data for future research.

Due to uncertainty in the exact problem generation from the *Turing.co* paper,  
**indirect comparisons** are made where identical input matrices are unavailable.

---

## 📂 Repository Contents
- **`problems/`** — Scripts for generating Max-Cut, NAE-3-SAT, and SK problem instances.
- **`lqa_bowles/`** — PyTorch implementation of LQA (adapted from Bowles et al.).
- **`benchmarking/`** — Benchmarking scripts for running LQA and other solvers.
- **`results/`** — Recorded metrics, plots, and comparative analysis.

cd lqa-benchmarking
pip install -r requirements.txt
