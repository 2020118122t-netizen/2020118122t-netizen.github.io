"""Step 2: paired p=1 grid search and common-basis benchmark.
Keep this file next to hw_qaoa_toffoli_step1.py. No IBM account needed.
"""
import csv
from pathlib import Path
from time import perf_counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from hw_qaoa_toffoli_step1 import make_qaoa_circuit

OUT = Path(__file__).resolve().parent / 'toffoli_step2_results'
METHODS = ('pauli', 'toffoli')
SEED = 42
GRID_SHOTS, FINAL_SHOTS = 2048, 8192
BASIS = ['rz', 'sx', 'x', 'cx']
REPEATS = 20


def save_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    sim = AerSimulator(method='statevector', max_parallel_threads=1, fusion_enable=False)
    gamma, beta = Parameter('gamma'), Parameter('beta')
    grid = [(float(g), float(b)) for g in np.linspace(0, 2*np.pi, 25)
            for b in np.linspace(0, np.pi, 25)]
    templates, grid_probs, grid_time, resources = {}, {}, {}, []
    print('QAOA p=1 | 625 grid points | 2048 shots each | seed=42', flush=True)
    for method in METHODS:
        logical = make_qaoa_circuit(gamma, beta, method, measure=True)
        # Same generic basis and no hardware routing; exclude compilation from run time.
        t0 = perf_counter()
        compiled = transpile(logical, basis_gates=BASIS, optimization_level=3,
                             seed_transpiler=SEED)
        compile_s = perf_counter() - t0
        templates[method] = compiled
        print(f'Grid search: {method} ...', flush=True)
        t0 = perf_counter()
        values = []
        for start in range(0, len(grid), 50):
            batch = [compiled.assign_parameters({gamma:g, beta:b}) for g,b in grid[start:start+50]]
            result = sim.run(batch, shots=GRID_SHOTS, seed_simulator=SEED+start).result()
            values.extend(result.get_counts(i).get('111', 0)/GRID_SHOTS for i in range(len(batch)))
        grid_time[method] = perf_counter() - t0
        grid_probs[method] = np.array(values)
        ops = compiled.count_ops()
        resources.append(dict(method=method, qubits=logical.num_qubits,
            logical_gates=sum(v for k,v in logical.count_ops().items() if k not in ('barrier','measure')),
            logical_depth=logical.depth(), basis_gates=sum(v for k,v in ops.items() if k not in ('barrier','measure')),
            basis_depth=compiled.depth(), basis_cx=ops.get('cx',0), compile_s=compile_s,
            grid_wall_s=grid_time[method]))
        print(f'Finished {method}: grid wall time {grid_time[method]:.3f} s', flush=True)
    # Select shared angles based on the Pauli reference search, not independently.
    idx = int(np.argmax(grid_probs['pauli']))
    g,b = grid[idx]
    print(f'\nShared best grid angles: gamma={g:.8f}, beta={b:.8f}')
    rows = [dict(gamma=gg,beta=bb,pauli_p111=grid_probs['pauli'][i],
                 toffoli_p111=grid_probs['toffoli'][i]) for i,(gg,bb) in enumerate(grid)]
    save_csv('grid_search.csv',rows)
    final_circuits, exact, counts_all = {}, {}, {}
    for method in METHODS:
        final_circuits[method] = templates[method].assign_parameters({gamma:g,beta:b})
        sv = Statevector.from_instruction(make_qaoa_circuit(g,b,method,measure=False))
        exact[method] = sv.probabilities(qargs=[0,1,2])
        if method == 'toffoli':
            assert np.isclose(sv.probabilities(qargs=[3,4])[0],1.0,atol=1e-10)
        counts_all[method] = sim.run(final_circuits[method],shots=FINAL_SHOTS,seed_simulator=2026).result().get_counts()
        (OUT / f'{method}_compiled_circuit.txt').write_text(str(final_circuits[method].draw('text',fold=-1)),encoding='utf-8')
    assert np.allclose(exact['pauli'],exact['toffoli'],atol=1e-10)
    # Warm-up above; alternate order, reuse precompiled circuits; end-to-end local wall time.
    times = {m:[] for m in METHODS}
    for rep in range(REPEATS):
        for method in (METHODS if rep%2==0 else METHODS[::-1]):
            t0 = perf_counter()
            sim.run(final_circuits[method],shots=FINAL_SHOTS,seed_simulator=1000+rep).result()
            times[method].append(perf_counter()-t0)
    save_csv('timing_runs.csv',[dict(repeat=i+1,pauli_wall_s=times['pauli'][i],toffoli_wall_s=times['toffoli'][i]) for i in range(REPEATS)])
    for r in resources:
        m=r['method']; p=counts_all[m].get('111',0)/FINAL_SHOTS
        r.update(gamma=g,beta=b,grid_p111=float(grid_probs[m][idx]), exact_p111=float(exact[m][7]),
                 final_p111=p,expectation=-p,median_wall_ms=1000*float(np.median(times[m])))
        print('\n'+m.upper())
        for k,v in r.items(): print(f'  {k}: {v}')
        print('  counts:',counts_all[m])
    save_csv('comparison_summary.csv',resources)
    states=[f'{i:03b}' for i in range(8)]
    save_csv('final_counts.csv',[dict(state=s,pauli=counts_all['pauli'].get(s,0),toffoli=counts_all['toffoli'].get(s,0)) for s in states])
    fig,ax=plt.subplots(figsize=(8,4.5)); x=np.arange(8)
    for i,m in enumerate(METHODS): ax.bar(x+(i-.5)*.36,[counts_all[m].get(s,0)/FINAL_SHOTS for s in states],.36,label=m)
    ax.set(xticks=x,xticklabels=states,ylabel='Measurement probability',title='Optimized QAOA p=1 (8192 shots)'); ax.legend(); fig.tight_layout(); fig.savefig(OUT/'final_histogram.png',dpi=180); plt.close(fig)
    for m in METHODS:
        fig,ax=plt.subplots(figsize=(6,5)); im=ax.imshow(grid_probs[m].reshape(25,25).T,origin='lower',aspect='auto',extent=(0,2*np.pi,0,np.pi)); ax.scatter([g],[b],marker='x',color='white'); ax.set(xlabel='gamma',ylabel='beta',title=f'{m}: grid search P(111)'); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(OUT/f'{m}_grid.png',dpi=180); plt.close(fig)
    print('\nPASS: shared angles produce equivalent ideal probabilities.')
    print('Timing is local Aer wall time, not QPU runtime. Common basis has no hardware coupling map.')
    print('Results saved to:',OUT)

if __name__=='__main__': main()
