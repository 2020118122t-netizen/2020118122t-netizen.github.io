from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt


def make_qaoa_circuit(gamma, beta):
    qc = QuantumCircuit(3)

    # Initial superposition
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.barrier()

    # Cost Hamiltonian
    # Hc = -(1/8)(I-Z0)(I-Z1)(I-Z2)
    # Identity term is omitted because it only contributes global phase.

    qc.rz(gamma / 4, 0)
    qc.rz(gamma / 4, 1)
    qc.rz(gamma / 4, 2)

    qc.cx(0, 1)
    qc.rz(-gamma / 4, 1)
    qc.cx(0, 1)

    qc.cx(0, 2)
    qc.rz(-gamma / 4, 2)
    qc.cx(0, 2)

    qc.cx(1, 2)
    qc.rz(-gamma / 4, 2)
    qc.cx(1, 2)

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
    qc.measure_all()

    return qc


# Optimal parameters found with Aer
gamma = 2.0943951023931953
beta = 2.617993877991494

qc = make_qaoa_circuit(gamma, beta)

# IBM Quantum connection using locally saved credentials
service = QiskitRuntimeService()

backend = service.least_busy(
    operational=True,
    simulator=False,
    min_num_qubits=3
)

print("\n========================================")
print("       IBM QUANTUM BACKEND")
print("========================================")
print("Selected backend =", backend.name)
print("Number of qubits =", backend.num_qubits)

# Transpile to the selected IBM QPU topology/native gates
pm = generate_preset_pass_manager(
    backend=backend,
    optimization_level=3,
    seed_transpiler=42
)

isa_circuit = pm.run(qc)

print("\n========================================")
print("       TRANSPILED CIRCUIT")
print("========================================")
print("Circuit depth =", isa_circuit.depth())
print("Gate count =", isa_circuit.count_ops())

try:
    selected_qubits = isa_circuit.layout.final_index_layout(filter_ancillas=True)
    print("Selected physical qubits =", selected_qubits)
except Exception:
    print("Layout =", isa_circuit.layout)

ops = isa_circuit.count_ops()
two_qubit_gate_count = (
    ops.get("cx", 0)
    + ops.get("cz", 0)
    + ops.get("ecr", 0)
)
print("2-qubit gate count =", two_qubit_gate_count)

# Real IBM QPU run
shots = 4096
sampler = Sampler(mode=backend)
job = sampler.run([isa_circuit], shots=shots)

print("\n========================================")
print("       IBM JOB SUBMITTED")
print("========================================")
print("Job ID =", job.job_id())
print("Backend =", backend.name)
print("\nWaiting for IBM Quantum result...")

result = job.result()
pub_result = result[0]
counts = pub_result.data.meas.get_counts()

p111 = counts.get("111", 0) / shots
objective_expectation = -p111
most_frequent_state = max(counts, key=counts.get)

print("\n========================================")
print("       IBM QPU RESULT")
print("========================================")
print("Counts =", counts)
print("\nMost frequent bitstring =", most_frequent_state)
print("P(111) =", p111)
print("Objective expectation =", objective_expectation)
print("Shots =", shots)

# Histogram
fig = plot_histogram(
    counts,
    title=f"QAOA IBM Quantum Result ({backend.name})"
)
fig.savefig("ibm_qaoa_histogram.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nHistogram saved: ibm_qaoa_histogram.png")
print("\n========================================")
print("       IBM QPU RUN COMPLETE")
print("========================================")
