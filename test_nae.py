import torch
import random
import time
import numpy as np
import dimod
import simulated_bifurcation as sb
import gurobipy as gp
from gurobipy import GRB
from math import floor
from lqa_basic import Lqa_basic
import cim_optimizer.solve_Ising as cim

# --- Generate NAE-3-SAT Ising instance ---
def generate_nae3sat_ising(n_vars, m_clauses):
    """
    Generates couplings matrix J and constant term C for NAE-3-SAT mapped 
    to an Ising model from the Turing paper:
    
    H(σ) = 1/4 * Σ_m [ ζ_m1 ζ_m2 σ_i1 σ_i2 +
                       ζ_m2 ζ_m3 σ_i2 σ_i3 +
                       ζ_m3 ζ_m1 σ_i3 σ_i1 + 1 ]
    """
    J = np.zeros((n_vars, n_vars))
    constant_term = 0.0

    for _ in range(m_clauses):
        i1, i2, i3 = random.sample(range(n_vars), 3)
        zeta = [random.choice([-1, 1]) for _ in range(3)]

        J[i1, i2] += zeta[0] * zeta[1] / 4
        J[i2, i3] += zeta[1] * zeta[2] / 4
        J[i3, i1] += zeta[2] * zeta[0] / 4

        # Symmetrize
        J[i2, i1] = J[i1, i2]
        J[i3, i2] = J[i2, i3]
        J[i1, i3] = J[i3, i1]

        constant_term += 1/4  # "+1" term in Hamiltonian

    return torch.tensor(J, dtype=torch.float32), constant_term

# --- Prepare dimod input ---
def prepare_dimod_input(J_torch):
    J_np = J_torch.numpy()
    n = J_np.shape[0]
    J_dict = {}
    for i in range(n):
        for j in range(i + 1, n):
            if J_np[i, j] != 0:
                J_dict[(i, j)] = float(J_np[i, j])
    h = {i: 0.0 for i in range(n)}
    return h, J_dict

# --- Solvers ---
def solve_dimod(h, J_dict):
    bqm = dimod.BinaryQuadraticModel(h, J_dict, 0.0, vartype=dimod.SPIN)
    sampler = dimod.SimulatedAnnealingSampler()
    start = time.time()
    sampleset = sampler.sample(bqm, num_reads=15)
    elapsed = time.time() - start
    best_energy = sampleset.first.energy
    return best_energy, elapsed

def solve_cim(J_np):
    start = time.time()
    solution = cim.Ising(J_np).solve(hyperparameters_randomtune=False)
    elapsed = time.time() - start
    best_energy = solution.result['lowest_energy']
    return best_energy, elapsed

def solve_simulated_bifurcation(J_np):
    start = time.time()
    ising = sb.QuadraticPolynomial(J_np / 2)
    spins, value = ising.minimize(domain='spin')
    elapsed = time.time() - start
    return value.item(), elapsed

def solve_gurobi(J_np):
    n = J_np.shape[0]
    model = gp.Model()
    model.setParam('TimeLimit', 1)
    model.setParam("OutputFlag", 0)
    model.Params.LogToConsole = 0  # silence output

    # Variables: s_i in {-1, +1}
    spins = model.addVars(n, vtype=GRB.INTEGER, lb=-1, ub=1, name="s")

    # Add quadratic objective: sum_{i<j} J_ij * s_i * s_j
    obj = gp.QuadExpr()
    for i in range(n):
        for j in range(i+1, n):
            if J_np[i, j] != 0:
                obj.add(J_np[i, j] * spins[i] * spins[j])
    model.setObjective(obj, GRB.MINIMIZE)

    # s_i in {-1, +1} integer constraints:
    for i in range(n):
        model.addConstr(spins[i] * spins[i] == 1)

    start = time.time()
    model.optimize()  # silence output after optimization
    elapsed = time.time() - start

    if model.status == GRB.OPTIMAL:
        best_energy = model.objVal
    else:
        best_energy = None
    return best_energy, elapsed

# --- LQA solver run ---
def LQA(J):
    print(f"\nRunning LQA solver on the provided Ising instance...")

    lqa_solver = Lqa_basic(J)
    start_time = time.time()
    lqa_solver.minimise(step=0.5, N=1000, g=1, f=0.1, mom=0.99)
    elapsed = time.time() - start_time

    min_energy = lqa_solver.energy  # get energy from attribute

    print(f"Min energy (LQA): {min_energy}")
    print(f"Time taken (LQA): {elapsed:.4f} seconds")

    return min_energy, elapsed

# --- Main benchmarking run ---
if __name__ == "__main__":
    #n_vars = range(100,1000,100)
    for n_vars in range(100, 1001, 100):
        m_clauses = int(2.11 * n_vars)
        print(f"Generating NAE-3-SAT instance with {n_vars} variables and {m_clauses} clauses...")
        J, constant_term = generate_nae3sat_ising(n_vars, m_clauses)
        J_np = J.numpy()

        h, J_dict = prepare_dimod_input(J)

        # Run all other solvers
        dimod_energy, dimod_time = solve_dimod(h, J_dict)
        cim_energy, cim_time = solve_cim(J_np)
        sb_energy, sb_time = solve_simulated_bifurcation(J_np)
        gurobi_energy, gurobi_time = solve_gurobi(J_np)
        lqa_energy, lqa_time = LQA(J)

        # Final summary
        print("\n===== Energy Summary =====")
        print(f"Dimod Simulated Annealing Energy: {dimod_energy}")
        print(f"CIM Solver Energy: {cim_energy}")
        print(f"Simulated Bifurcation Energy: {sb_energy}")
        print(f"Gurobi Energy: {gurobi_energy if gurobi_energy is not None else 'N/A'}")
        print(f"LQA Solver Energy: {lqa_energy}")
