"""Step 4: six QAOA circuits on one IBM QPU (SamplerV2).
Submit once: python hw_qaoa_toffoli_step4.py --submit
Retrieve existing job: python hw_qaoa_toffoli_step4.py --resume
Requires saved IBM account, step3 script, qiskit-ibm-runtime and step3 dependencies.
"""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qiskit import qpy
from qiskit.circuit import ParameterVector
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from hw_qaoa_toffoli_step3 import circuit, exact_prob, METHODS, LABELS, P1

OUT = Path(__file__).resolve().parent / 'toffoli_step4_results'
MANIFEST = OUT / 'job_manifest.json'
SHOTS = 8192
SEED = 42
# Step 3 values, rounded only at the displayed eight decimal places.
ANGLES = {1:P1, 2:np.array([4.41541059,4.04824507,0.69583252,0.45351484])}
AER = {(1,'pauli'):0.4232177734375,(1,'toffoli_1anc'):0.425903,
       (1,'toffoli_2anc'):0.4197998046875,(2,'pauli'):0.744995,
       (2,'toffoli_1anc'):0.739990,(2,'toffoli_2anc'):0.736206}


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')


def save_csv(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def prepare(backend):
    pm=generate_preset_pass_manager(backend=backend,optimization_level=3,seed_transpiler=SEED)
    rows=[]; circuits=[]
    for p,theta in ANGLES.items():
        gs,bs=ParameterVector('gamma',p),ParameterVector('beta',p)
        bind={**dict(zip(gs,theta[:p])),**dict(zip(bs,theta[p:]))}
        for method in METHODS:
            logical=circuit(gs,bs,method)
            isa=pm.run(logical).assign_parameters(bind)
            ops=isa.count_ops()
            gates=sum(1 for inst in isa.data if inst.operation.name not in ('barrier','measure','delay'))
            twoq=sum(1 for inst in isa.data if len(inst.qubits)==2 and inst.operation.name not in ('barrier','measure','delay'))
            active={isa.find_bit(q).index for inst in isa.data if inst.operation.name not in ('barrier','delay') for q in inst.qubits}
            exact=float(exact_prob(theta,method)[0][7])
            row=dict(p=p,method=method,logical_qubits=logical.num_qubits,active_physical_qubits=len(active),
                     physical_qubit_indices=';'.join(map(str,sorted(active))),isa_gates=gates,isa_depth=isa.depth(),
                     isa_two_qubit_gates=twoq,isa_operations=json.dumps(dict(ops)),gamma=';'.join(map(str,theta[:p])),
                     beta=';'.join(map(str,theta[p:])),exact_p111=exact,aer_p111=AER[p,method],register=isa.cregs[0].name)
            rows.append(row); circuits.append(isa)
            print(f'p={p} {method:14s} ISA gates={gates} depth={isa.depth()} 2Q={twoq} active={len(active)}',flush=True)
    return circuits,rows


def collect(job,manifest):
    print('Job ID:',job.job_id(),flush=True)
    print('Waiting for IBM job result (queue time is not QPU execution time).',flush=True)
    t=perf_counter(); result=job.result(); wait_s=perf_counter()-t
    rows=[]; count_rows=[]; pub_meta=[]
    if len(result)!=len(manifest['rows']): raise RuntimeError('Unexpected number of PUB results')
    for i,base in enumerate(manifest['rows']):
        pub=result[i]; counts=getattr(pub.data,base['register']).get_counts()
        total=sum(counts.values()); success=counts.get('111',0)/total
        row=dict(base); row.update(backend=manifest['backend'],job_id=job.job_id(),shots=total,qpu_p111=success,
                                   expectation=-success,loss_vs_ideal_pp=100*(base['exact_p111']-success))
        rows.append(row); pub_meta.append(pub.metadata)
        for state in range(8):
            s=f'{state:03b}'; count_rows.append(dict(p=base['p'],method=base['method'],state=s,count=counts.get(s,0)))
        print(f"p={base['p']} {base['method']:14s} ideal={base['exact_p111']:.6f} Aer={base['aer_p111']:.6f} QPU={success:.6f} shots={total}")
    save_csv('qpu_comparison.csv',rows); save_csv('qpu_counts.csv',count_rows)
    dump(OUT/'pub_metadata.json',pub_meta)
    try: metrics=job.metrics()
    except Exception as exc: metrics={'unavailable':str(exc)}
    dump(OUT/'job_metrics.json',metrics)
    dump(OUT/'retrieval_info.json',{'job_id':job.job_id(),'result_retrieval_wall_s':wait_s,'backend':manifest['backend']})
    fig,ax=plt.subplots(figsize=(10,5)); x=np.arange(len(rows)); w=.25
    for i,(key,label) in enumerate([('exact_p111','Ideal'),('aer_p111','Aer'),('qpu_p111','IBM QPU')]):
        ax.bar(x+(i-1)*w,[100*r[key] for r in rows],w,label=label)
    ax.set_xticks(x,[f"p={r['p']}\n{LABELS[METHODS.index(r['method'])]}" for r in rows],rotation=15,ha='right')
    ax.set(ylabel='P(111) (%)',ylim=(0,100),title='Ideal vs Aer vs IBM QPU'); ax.legend(); fig.tight_layout()
    fig.savefig(OUT/'qpu_success_comparison.png',dpi=180); plt.close(fig)
    print('Job metrics:',metrics)
    print('Job-level usage is not per-circuit QPU runtime. Physical layouts may differ.')
    print('Results saved to:',OUT)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--submit',action='store_true'); group.add_argument('--resume',action='store_true')
    parser.add_argument('--backend',help='Optional IBM backend name; otherwise least busy operational QPU')
    args=parser.parse_args(); OUT.mkdir(exist_ok=True)
    if args.submit and MANIFEST.exists():
        raise SystemExit('Existing job manifest found. Use --resume to avoid duplicate QPU usage.')
    if args.resume and not MANIFEST.exists(): raise SystemExit('No existing job manifest found.')
    service=QiskitRuntimeService()
    if args.resume:
        manifest=json.loads(MANIFEST.read_text(encoding='utf-8')); collect(service.job(manifest['job_id']),manifest); return
    backend=service.backend(args.backend) if args.backend else service.least_busy(operational=True,simulator=False,min_num_qubits=5)
    print('IBM backend:',backend.name,flush=True)
    circuits,rows=prepare(backend)
    save_csv('isa_resources.csv',rows)
    with (OUT/'isa_circuits.qpy').open('wb') as f: qpy.dump(circuits,f)
    sampler=SamplerV2(mode=backend)
    print(f'Submitting ONE job, {len(circuits)} circuits x {SHOTS} shots. This uses IBM QPU allocation.',flush=True)
    job=sampler.run(circuits,shots=SHOTS)
    manifest={'job_id':job.job_id(),'backend':backend.name,'shots_per_circuit':SHOTS,'rows':rows}
    dump(MANIFEST,manifest)
    collect(job,manifest)

if __name__=='__main__': main()
