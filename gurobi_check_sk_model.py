import numpy as np
import torch
from gurobipy import Model, GRB

def generate_sk_matrix(N, seed):
    np.random.seed(seed)
    rand_vals = np.random.randn(N, N)
    J = torch.tensor(rand_vals, dtype=torch.float32)
    J = torch.triu(J, diagonal=1)
    J = J + J.T  # Make symmetric
    J.fill_diagonal_(0)
    return J.numpy()

def solve_sk_gurobi(J):
    n = J.shape[0]
    model = Model("SK_Model")
    model.setParam('TimeLimit', 1)
    model.setParam("OutputFlag", 0)

    # Binary variables x_i ∈ {0, 1}
    x = [model.addVar(vtype=GRB.BINARY, name=f"x_{i}") for i in range(n)]

    # Objective: maximize ∑ J_ij * s_i * s_j, where s_i = 2x_i - 1
    obj = 0
    for i in range(n):
        for j in range(i + 1, n):
            Jij = J[i, j]
            term = Jij * (4 * x[i] * x[j] - 2 * x[i] - 2 * x[j] + 1)
            obj += term

    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    # Convert binary solution to spins
    binary_solution = np.array([int(var.X) for var in x])
    spins = 2 * binary_solution - 1
    energy = np.sum(J * np.outer(spins, spins)) / 2

    return spins, energy

def main():
    N = 128
    seeds = range(10) 
    for seed in seeds:# <-- Change this seed as needed
        J = generate_sk_matrix(N, seed)
        spins, energy = solve_sk_gurobi(J)
        print(f"Seed {seed} | Energy: {energy:.4f}")
       # print(f"Spin configuration:\n{spins}")

if __name__ == "__main__":
    main()
