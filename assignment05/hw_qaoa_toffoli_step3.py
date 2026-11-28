"""Step 3: p=1/2, Pauli vs Toffoli with one/two clean ancillas.
Run: python hw_qaoa_toffoli_step3.py
Requires qiskit, qiskit-aer, scipy, numpy, matplotlib. No IBM account.
"""
import csv
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

OUT = Path(__file__).resolve().parent / 'toffoli_step3_results'
METHODS = ('pauli', 'toffoli_1anc', 'toffoli_2anc')
LABELS = ('Pauli-Z', 'Toffoli + CP (1 anc)', 'Toffoli + P (2 anc)')
BASIS = ['rz', 'sx', 'x', 'cx']
SEED, SHOTS, REPEATS = 42, 8192, 20
P1 = np.array([2*np.pi/3, 5*np.pi/6]) # shared Step 2 angles


def circuit(gammas, betas, method, measure=True):
    n = {'pauli':3, 'toffoli_1anc':4, 'toffoli_2anc':5}[method]
    qc = QuantumCircuit(n, 3) if measure else QuantumCircuit(n)
    qc.h([0,1,2]); qc.barrier()
    for g,b in zip(gammas,betas):
        if method == 'pauli':
            for q in range(3): qc.rz(g/4,q)
            for i,j in ((0,1),(0,2),(1,2)):
                qc.cx(i,j); qc.rz(-g/4,j); qc.cx(i,j)
            qc.cx(0,2); qc.cx(1,2); qc.rz(g/4,2)
            qc.cx(1,2); qc.cx(0,2)
        elif method == 'toffoli_1anc':
            qc.ccx(0,1,3); qc.cp(g,3,2); qc.ccx(0,1,3)
        else:
            qc.ccx(0,1,3); qc.ccx(3,2,4); qc.p(g,4)
            qc.ccx(3,2,4); qc.ccx(0,1,3)
        qc.barrier(); qc.rx(2*b,[0,1,2]); qc.barrier()
    if measure: qc.measure([0,1,2],[0,1,2])
    return qc


def save_csv(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def exact_prob(theta, method):
    p=len(theta)//2
    sv=Statevector.from_instruction(circuit(theta[:p],theta[p:],method,False))
    anc=1.0 if method=='pauli' else float(sv.probabilities(qargs=list(range(3,sv.num_qubits)))[0])
    return sv.probabilities(qargs=[0,1,2]),anc


def optimize_p2():
    # Exact reference optimization only, with two independent gammas and betas.
    # p=1 is retained from Step 2; p=2 uses a different optimization budget.
    rng=np.random.default_rng(SEED)
    embedded=np.array([P1[0],0.0,P1[1],0.0])
    starts=[embedded]+[rng.uniform(0,[2*np.pi,2*np.pi,np.pi,np.pi]) for _ in range(7)]
    best=embedded.copy(); best_p=float(exact_prob(best,'pauli')[0][7]); history=[]
    def obj(t): return -float(exact_prob(t,'pauli')[0][7])
    t0=perf_counter()
    for i,start in enumerate(starts):
        res=minimize(obj,start,method='COBYLA',options={'maxiter':200,'rhobeg':0.5,'tol':1e-7})
        score=-float(res.fun)
        if score>best_p: best_p=score; best=res.x.copy()
        history.append(dict(start=i+1,evaluations=res.nfev,p111=score,success=bool(res.success)))
        print(f'p=2 start {i+1}/{len(starts)}: P(111)={score:.8f}',flush=True)
    elapsed=perf_counter()-t0
    best[:2]%=2*np.pi; best[2:]%=np.pi
    save_csv('p2_optimizer_runs.csv',history)
    print(f'p=2 exact reference search wall time: {elapsed:.3f} s (not QPU time)')
    return best


def main():
    OUT.mkdir(exist_ok=True)
    sim=AerSimulator(method='statevector',max_parallel_threads=1,fusion_enable=False)
    print('STEP 3: p=1 vs p=2 | 3 cost implementations | common basis, no routing',flush=True)
    angles={1:P1,2:optimize_p2()}
    rows=[]; compiled={}; counts={}; timing={}; exact={}
    for p,theta in angles.items():
        print(f'\np={p} gamma={theta[:p]} beta={theta[p:]}',flush=True)
        gs,bs=ParameterVector('gamma',p),ParameterVector('beta',p)
        ref,_=exact_prob(theta,'pauli')
        for method in METHODS:
            key=(p,method)
            logical=circuit(gs,bs,method)
            basis=transpile(logical,basis_gates=BASIS,optimization_level=3,seed_transpiler=SEED)
            bind={**dict(zip(gs,theta[:p])),**dict(zip(bs,theta[p:]))}
            qc=basis.assign_parameters(bind); compiled[key]=qc
            probs,anc=exact_prob(theta,method); exact[key]=float(probs[7])
            assert np.allclose(probs,ref,atol=1e-10),key
            assert np.isclose(anc,1,atol=1e-10),key
            # Verify compiled distribution too (before the final measurements).
            compiled_sv=Statevector.from_instruction(qc.remove_final_measurements(inplace=False))
            assert np.allclose(compiled_sv.probabilities(qargs=[0,1,2]),ref,atol=1e-10),key
            counts[key]=sim.run(qc,shots=SHOTS,seed_simulator=2026).result().get_counts()
            sample=counts[key].get('111',0)/SHOTS
            gates=lambda c: sum(v for k,v in c.count_ops().items() if k not in ('barrier','measure'))
            rows.append(dict(p=p,method=method,qubits=logical.num_qubits,parameters=2*p,
                logical_gates=gates(logical),logical_depth=logical.depth(),logical_ccx=logical.count_ops().get('ccx',0),
                logical_p=logical.count_ops().get('p',0),logical_cp=logical.count_ops().get('cp',0),
                basis_gates=gates(basis),basis_depth=basis.depth(),basis_cx=basis.count_ops().get('cx',0),
                gamma=';'.join(map(str,theta[:p])),beta=';'.join(map(str,theta[p:])),
                exact_p111=exact[key],sample_p111=sample,expectation=-sample,ancilla_zero_probability=anc))
            (OUT/f'p{p}_{method}_logical.txt').write_text(str(logical.assign_parameters(bind).draw('text',fold=-1)),encoding='utf-8')
            (OUT/f'p{p}_{method}_compiled.txt').write_text(str(qc.draw('text',fold=-1)),encoding='utf-8')
    keys=list(compiled); timing={k:[] for k in keys}
    # All configurations warmed above. Rotate order to reduce systematic bias.
    for r in range(REPEATS):
        for key in keys[r%len(keys):]+keys[:r%len(keys)]:
            t0=perf_counter(); sim.run(compiled[key],shots=SHOTS,seed_simulator=1000+r).result()
            timing[key].append(1000*(perf_counter()-t0))
    for row in rows:
        key=(row['p'],row['method']); row['median_wall_ms']=float(np.median(timing[key])); row['q1_wall_ms']=float(np.percentile(timing[key],25)); row['q3_wall_ms']=float(np.percentile(timing[key],75))
        print(f"p={row['p']} {row['method']:14s} qubits={row['qubits']} logical={row['logical_gates']}/{row['logical_depth']} basis={row['basis_gates']}/{row['basis_depth']} CX={row['basis_cx']} exact={row['exact_p111']:.6f} sampled={row['sample_p111']:.6f} median={row['median_wall_ms']:.3f} ms")
    save_csv('comparison_summary.csv',rows)
    save_csv('timing_runs.csv',[dict(p=p,method=m,repeat=r+1,wall_ms=timing[p,m][r]) for p,m in keys for r in range(REPEATS)])
    states=[f'{i:03b}' for i in range(8)]
    save_csv('final_counts.csv',[dict(p=p,method=m,state=s,count=counts[p,m].get(s,0)) for p,m in keys for s in states])
    for metric,title in [('logical_depth','Logical circuit depth'),('basis_depth','Common-basis circuit depth'),('basis_cx','Common-basis CX count'),('median_wall_ms','Aer wall time (ms)'),('sample_p111','Sampled P(111)')]:
        fig,ax=plt.subplots(figsize=(8,4.5)); x=np.arange(2)
        for i,m in enumerate(METHODS):
            vals=[next(r[metric] for r in rows if r['p']==p and r['method']==m) for p in (1,2)]
            ax.bar(x+(i-1)*.24,vals,.24,label=LABELS[i])
        ax.set(xticks=x,xticklabels=['p=1','p=2'],ylabel=title,title=title); ax.legend(); fig.tight_layout(); fig.savefig(OUT/f'{metric}.png',dpi=180); plt.close(fig)
    print('\nPASS: equivalent ideal distributions and restored ancillas for all configurations.')
    print('p=1 uses Step 2 grid angles; p=2 uses exact multistart COBYLA. Different parameter-search budgets; not an equal-budget comparison across p.')
    print('Timing is local Aer wall time, not QPU runtime. No hardware coupling map.')
    print('Results saved to:',OUT)

if __name__=='__main__': main()
