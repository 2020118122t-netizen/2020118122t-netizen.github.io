"""QAOA homework: step 1/2, Pauli vs Toffoli circuit smoke test.
Objective: minimize -x0*x1*x2.  Only data qubits are measured.
Run: python hw_qaoa_toffoli_step1.py
Requires: qiskit, qiskit-aer, matplotlib, numpy
"""
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

GAMMA = 1.0
BETA = 0.5
SHOTS = 4096
SEED = 42
OUT = Path(__file__).resolve().parent / 'toffoli_step1_results'


def make_qaoa_circuit(gamma, beta, method='toffoli', p=1, measure=True):
    """Scalar angles are reused across p layers, for this initial test."""
    if method not in ('pauli', 'toffoli'):
        raise ValueError('method must be pauli or toffoli')
    n = 3 if method == 'pauli' else 5
    qc = QuantumCircuit(n, 3) if measure else QuantumCircuit(n)
    qc.h([0, 1, 2])
    qc.barrier()
    for _ in range(p):
        if method == 'pauli':
            # Original homework: omit the identity term (global phase).
            for q in range(3):
                qc.rz(gamma / 4, q)
            for i, j in [(0, 1), (0, 2), (1, 2)]:
                qc.cx(i, j)
                qc.rz(-gamma / 4, j)
                qc.cx(i, j)
            qc.cx(0, 2)
            qc.cx(1, 2)
            qc.rz(gamma / 4, 2)
            qc.cx(1, 2)
            qc.cx(0, 2)
        else:
            # Compute x0*x1, then x0*x1*x2 into clean ancillas.
            qc.ccx(0, 1, 3)
            qc.ccx(3, 2, 4)
            # Hc = -x0*x1*x2, so Uc = exp(+i*gamma*x0*x1*x2).
            qc.p(gamma, 4)
            # Uncompute in reverse order; ancillas return to |00>.
            qc.ccx(3, 2, 4)
            qc.ccx(0, 1, 3)
        qc.barrier()
        qc.rx(2 * beta, [0, 1, 2])
        qc.barrier()
    if measure:
        qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def main():
    OUT.mkdir(exist_ok=True)
    simulator = AerSimulator()
    print('Objective: minimize -x0*x1*x2; optimum 111, f=-1')
    print(f'gamma={GAMMA}, beta={BETA}, shots={SHOTS}, seed={SEED}')
    circuits = {}
    probs = {}
    counts_all = {}
    for method in ('pauli', 'toffoli'):
        qc = make_qaoa_circuit(GAMMA, BETA, method)
        sv = Statevector.from_instruction(make_qaoa_circuit(GAMMA, BETA, method, measure=False))
        probs[method] = sv.probabilities(qargs=[0, 1, 2])
        if method == 'toffoli':
            anc = sv.probabilities(qargs=[3, 4])
            print('Ancilla P(00):', anc[0])
            assert np.isclose(anc[0], 1.0, atol=1e-10)
        compiled = transpile(qc, simulator, seed_transpiler=SEED)
        counts = simulator.run(compiled, shots=SHOTS, seed_simulator=SEED).result().get_counts()
        counts_all[method] = counts
        circuits[method] = qc
        p111 = counts.get('111', 0) / SHOTS
        print('\n' + '='*55)
        print(method.upper(), 'QAOA CIRCUIT')
        print(qc.draw('text'))
        print('Qubits:', qc.num_qubits)
        print('Logical gate counts:', dict(qc.count_ops()))
        print('Logical depth:', qc.depth())
        print('Measurement counts:', counts)
        print('Exact P(111):', probs[method][7])
        print('Sampled P(111):', p111)
        print('Objective expectation:', -p111)
        (OUT / f'{method}_circuit.txt').write_text(str(qc.draw('text', fold=-1)), encoding='utf-8')
    delta = np.max(np.abs(probs['pauli'] - probs['toffoli']))
    print('\nMaximum exact probability difference:', delta)
    assert np.allclose(probs['pauli'], probs['toffoli'], atol=1e-10)
    print('PASS: both circuits have the same ideal measurement distribution.')
    fig = plot_histogram([counts_all['pauli'], counts_all['toffoli']], legend=['Pauli-Z', 'Toffoli + P'], title='QAOA initial test (4096 shots)')
    fig.savefig(OUT / 'initial_histogram.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Results saved to:', OUT)


if __name__ == '__main__':
    main()
