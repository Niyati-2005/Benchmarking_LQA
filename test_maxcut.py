from lqa_basic import Lqa_basic
import torch
import time
import dimod
import simulated_bifurcation as sb
import cim_optimizer.solve_Ising as cim
import pandas as pd


def gset_file_to_matrix(filepath):
    """
    Converts a GSet file with 'n m' and 'i j w' format into a symmetric PyTorch tensor.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # First line: n (nodes) m (edges)
    n, m = map(int, lines[0].strip().split())
    J = torch.zeros((n, n))

    for line in lines[1:]:
        i, j, w = map(int, line.strip().split())
        i -= 1
        j -= 1
        J[i, j] = w
        J[j, i] = w  # symmetry

    J.fill_diagonal_(0)
    return J


def obj_maxcut_from_matrix(spins, coupling_matrix):
    """Compute Max-Cut value from spin configuration."""
    obj = 0.0
    n = len(spins)
    for i in range(n):
        for j in range(i + 1, n):
            if spins[i] != spins[j]:
                obj += coupling_matrix[i, j].item()
    return obj


def prepare_dimod_input(J_torch):
    J_np = J_torch.numpy()
    n = J_np.shape[0]
    J_dict = {(i, j): float(J_np[i, j])
              for i in range(n) for j in range(i + 1, n) if J_np[i, j] != 0}
    h = {i: 0.0 for i in range(n)}
    return h, J_dict


def solve_dimod(h, J_dict):
    bqm = dimod.BinaryQuadraticModel(h, J_dict, 0.0, vartype=dimod.SPIN)
    sampler = dimod.SimulatedAnnealingSampler()
    start = time.time()
    sampleset = sampler.sample(bqm, num_reads=15)
    elapsed = time.time() - start

    best_sample = sampleset.first.sample
    spins = torch.tensor([best_sample[i] for i in range(len(best_sample))])
    best_energy = sampleset.first.energy
    return spins, best_energy, elapsed


def solve_cim(J_np):
    start = time.time()
    solution = cim.Ising(J_np).solve(hyperparameters_randomtune=False)
    elapsed = time.time() - start
    spins = torch.tensor(solution.result['lowest_energy_spin_config'])
    best_energy = solution.result['lowest_energy']
    return spins, best_energy, elapsed


def solve_simulated_bifurcation(J_np):
    start = time.time()
    ising = sb.QuadraticPolynomial(J_np / 2)
    spins, value = ising.minimize(domain='spin')
    elapsed = time.time() - start

    spins = torch.tensor(spins)
    return spins, float(value), elapsed


def Lqa_solver(J):
    lqa_basic_solver = Lqa_basic(J)
    lqa_basic_solver.minimise(step=0.1, N=1000, g=1, f=0.1, mom=0.99)
    lqa_basic_solution = lqa_basic_solver.config.view(-1)
    cut_value = obj_maxcut_from_matrix(lqa_basic_solution, J)
    return lqa_basic_solver, cut_value


def main():
    filepath = r"C:\Niyati\Coding\Benchmarking_lqa_test\gset_instances\G72.txt"
    couplings = gset_file_to_matrix(filepath)
    #print("Loaded coupling matrix with shape:", couplings.shape)

    results = []

    # LQA
    lqa_basic_solver, lqa_cut_value = Lqa_solver(couplings)
    results.append({
        "Solver": "LQA",
        "Energy": lqa_basic_solver.energy,
        "Cut Value": lqa_cut_value,
        "Time (s)": lqa_basic_solver.opt_time
    })

    # DIMOD
    h, J_dict = prepare_dimod_input(couplings)
    dimod_spins, dimod_energy, dimod_time = solve_dimod(h, J_dict)
    #print("Dimod spins:", dimod_spins)
    dimod_cut_value = obj_maxcut_from_matrix(dimod_spins, couplings)
    results.append({
        "Solver": "Dimod Simulated Annealing",
        "Energy": dimod_energy,
        "Cut Value": dimod_cut_value,
        "Time (s)": dimod_time
    })

    # CIM
    couplingcim=couplings.numpy()*-1  # CIM expects negative couplings for max-cut
    cim_spins, cim_energy, cim_time = solve_cim(couplingcim)
    #print("CIM spins:", cim_spins)
    cim_cut_value = -(obj_maxcut_from_matrix(cim_spins, couplingcim))
    results.append({
        "Solver": "CIM",
        "Energy": cim_energy,
        "Cut Value": cim_cut_value,
        "Time (s)": cim_time
    })

    # Simulated Bifurcation
    sb_spins, sb_energy, sb_time = solve_simulated_bifurcation(couplings.numpy())
    sb_cut_value = obj_maxcut_from_matrix(sb_spins, couplings)
    results.append({
        "Solver": "Simulated Bifurcation",
        "Energy": sb_energy,
        "Cut Value": sb_cut_value,
        "Time (s)": sb_time
    })

    df = pd.DataFrame(results)
    best_cut = df["Cut Value"].max()
    df["Best?"] = df["Cut Value"] == best_cut

    print("\n===== Final Summary =====")
    print(df.to_string(index=False))
    df.to_csv("benchmark_results.csv", index=False)
    print("\nResults saved to benchmark_results.csv")


if __name__ == "__main__":
    main()
