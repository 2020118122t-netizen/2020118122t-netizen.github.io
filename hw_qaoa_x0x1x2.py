import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt


def make_qaoa_circuit(gamma, beta):
    qc = QuantumCircuit(3, 3)

    # Initial superposition
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.barrier()

    # Cost Hamiltonian
    # Hc = -(1/8)(I-Z0)(I-Z1)(I-Z2)
    # Identity term is omitted because it only contributes global phase.

    # Z terms
    qc.rz(gamma / 4, 0)
    qc.rz(gamma / 4, 1)
    qc.rz(gamma / 4, 2)

    # ZZ terms
    qc.cx(0, 1)
    qc.rz(-gamma / 4, 1)
    qc.cx(0, 1)

    qc.cx(0, 2)
    qc.rz(-gamma / 4, 2)
    qc.cx(0, 2)

    qc.cx(1, 2)
    qc.rz(-gamma / 4, 2)
    qc.cx(1, 2)

    # ZZZ term
    qc.cx(0, 2)
    qc.cx(1, 2)
    qc.rz(gamma / 4, 2)
    qc.cx(1, 2)
    qc.cx(0, 2)

    qc.barrier()

    # Mixer Hamiltonian
    qc.rx(2 * beta, 0)
    qc.rx(2 * beta, 1)
    qc.rx(2 * beta, 2)

    qc.barrier()
    qc.measure([0, 1, 2], [0, 1, 2])

    return qc


simulator = AerSimulator()

# Temporary test
Gamma_test = 1.0
Beta_test = 0.5
qc_test = make_qaoa_circuit(Gamma_test, Beta_test)

print("\n========================================")
print("       QAOA CIRCUIT TEST")
print("========================================")
print(qc_test.draw("text"))

qc_test_transpiled = transpile(qc_test, simulator)
test_shots = 4096

test_result = simulator.run(
    qc_test_transpiled,
    shots=test_shots
).result()

test_counts = test_result.get_counts()
test_p111 = test_counts.get("111", 0) / test_shots
test_expectation = -test_p111

print("\n========================================")
print("       TEST RESULT")
print("========================================")
print("gamma =", Gamma_test)
print("beta  =", Beta_test)
print("\nMeasurement Counts")
print(test_counts)
print("\nP(111) =", test_p111)
print("Objective expectation =", test_expectation)

# Grid search
print("\n========================================")
print("       GRID SEARCH START")
print("========================================")

gamma_list = np.linspace(0, 2 * np.pi, 25)
beta_list = np.linspace(0, np.pi, 25)

best_gamma = None
best_beta = None
best_p111 = -1

grid_shots = 2048

for gamma in gamma_list:
    for beta in beta_list:
        qc = make_qaoa_circuit(gamma, beta)
        qc_transpiled = transpile(qc, simulator)
        result = simulator.run(qc_transpiled, shots=grid_shots).result()
        counts = result.get_counts()
        p111 = counts.get("111", 0) / grid_shots

        if p111 > best_p111:
            best_p111 = p111
            best_gamma = gamma
            best_beta = beta

print("\n========================================")
print("       BEST PARAMETERS")
print("========================================")
print("Best gamma =", best_gamma)
print("Best beta =", best_beta)
print("Best P(111) =", best_p111)
print("Best objective expectation =", -best_p111)

# Final Aer run
print("\n========================================")
print("       FINAL AER RUN")
print("========================================")

final_qc = make_qaoa_circuit(best_gamma, best_beta)
final_qc_transpiled = transpile(final_qc, simulator)
final_shots = 8192

final_result = simulator.run(
    final_qc_transpiled,
    shots=final_shots
).result()

final_counts = final_result.get_counts()
final_p111 = final_counts.get("111", 0) / final_shots
final_expectation = -final_p111
most_frequent_state = max(final_counts, key=final_counts.get)

print("Optimal gamma =", best_gamma)
print("Optimal beta =", best_beta)
print("\nFinal Measurement Counts")
print(final_counts)
print("\nMost frequent bitstring =", most_frequent_state)
print("P(111) =", final_p111)
print("Objective expectation =", final_expectation)
print("\nCircuit depth =", final_qc_transpiled.depth())
print("Circuit gate count =", final_qc_transpiled.count_ops())

# Histogram
fig = plot_histogram(final_counts, title="QAOA Aer Simulator Result")
fig.savefig("aer_qaoa_histogram.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nHistogram saved: aer_qaoa_histogram.png")
print("\n========================================")
print("       AER SIMULATION COMPLETE")
print("========================================")
