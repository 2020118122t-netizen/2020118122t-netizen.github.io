"""
[과제 배경]
수업에서는 Binary Optimization 문제를 QUBO 형태로 만들고,
이를 gate-based quantum circuit으로 구성한 뒤 Classical 방식과
실행시간, 메모리, 해의 품질 등을 비교하여 Quantum Advantage가
있는지를 확인하는 방향으로 과제를 수행하였음

본 과제에서는 6-node Max-Cut 문제를 그대로 활용하였음. 같은 문제를
1) Classical Brute Force
2) Gate-based QAOA
두 방식으로 풀고 결과를 비교함

[실험 문제]
6-node Max-Cut
E = {(0,1), (1,2), (2,3), (3,4), (4,0), (4,5)}

각 node는 0 또는 1 그룹에 배정하며,
서로 다른 그룹을 연결하는 edge의 개수를 최대화한다.

Max-Cut은 최대화 문제이므로 QUBO에서는 부호를 바꾸어
Q(x) = -Cut(x)를 최소화함.

[비교 항목]
- Best / Exact Cut
- Mean Cut
- P(opt): 최적해가 측정될 확률
- Wall-clock execution time
- Peak RSS memory increase

[주의]
QAOA는 실제 quantum hardware가 아니라 Qiskit AerSimulator에서 실행하기에
따라서 이 실험의 runtime은 실제 quantum hardware speedup을 의미하지 않음,
"""

import csv
import itertools
import os
import threading
import time

import numpy as np
import psutil
from scipy.optimize import minimize

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter
    from qiskit_aer import AerSimulator
except ImportError as e:
    raise SystemExit(
        "\nQiskit/Aer가 설치되어 있지 않습니다.\n"
        "아래 명령을 먼저 실행하세요.\n\n"
        "pip install qiskit qiskit-aer scipy numpy psutil\n"
    ) from e


# ============================================================
# 0. 실험 설정
# ============================================================

N = 6

# 6-node Max-Cut graph
EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 0),
    (4, 5),
]

# QAOA 설정
SHOTS_TRAIN = 1024       # COBYLA가 angle을 찾을 때 objective 평가용 shots
SHOTS_FINAL = 8192       # 최종 parameter로 결과를 평가할 때 사용하는 shots
MAXITER = 50             # COBYLA 최대 반복 횟수
N_QAOA_RUNS = 5          # QAOA 독립 실행 횟수

# Classical은 너무 빨리 끝나므로 여러 번 반복한 평균 시간을 사용
CLASSICAL_REPEATS = 5000

# 같은 코드를 다시 실행했을 때 비슷한 결과가 나오도록 seed 고정
BASE_SEED = 20261107


# ============================================================
# 1. Max-Cut / QUBO 함수
# ============================================================

def cut_value(x):
    """
    Max-Cut 점수를 계산함

    edge 양 끝의 node 값이 다르면 1점,
    같으면 0점으로 계산함
    """
    return sum(int(x[i] != x[j]) for i, j in EDGES)


def qubo_value(x):
    """
    본 과제에서 사용하는 QUBO 목적함수.

    Q(x) = -Cut(x)

    Q(x) =
      -2(x0+x1+x2+x3) -3x4 -x5
      +2(x0x1+x1x2+x2x3+x3x4+x4x0+x4x5)

    QUBO는 최소화 문제이므로 Q 값이 작을수록 더 좋은 해임
    Exact Max-Cut = 5 이므로 QUBO minimum = -5임
    """
    x0, x1, x2, x3, x4, x5 = x

    return (
        -2 * (x0 + x1 + x2 + x3)
        - 3 * x4
        - x5
        + 2 * (
            x0 * x1
            + x1 * x2
            + x2 * x3
            + x3 * x4
            + x4 * x0
            + x4 * x5
        )
    )


# ============================================================
# 2. Classical Brute Force
# ============================================================

def classical_bruteforce():
    """
    가능한 2^6 = 64개의 비트열을 모두 확인

    따라서 이 방법은 근사해가 아니라 exact solution을 얻음
    문제 규모가 커지면 경우의 수가 2^N으로 증가하지만,
    N=6인 현재 문제에서는 매우 빠르게 계산함
    """
    best_q = float("inf")
    best_solutions = []

    for x in itertools.product([0, 1], repeat=N):
        q = qubo_value(x)

        if q < best_q:
            best_q = q
            best_solutions = [x]
        elif q == best_q:
            best_solutions.append(x)

    return best_q, best_solutions


def benchmark_classical_time(repeats=CLASSICAL_REPEATS):
    """
    Classical 1회 실행은 너무 짧아서 측정 오차가 클 수 있음
    따라서 여러 번 반복한 전체 시간을 측정한 뒤,
    반복 횟수로 나누어 1회 평균 실행시간을 계산함
    """
    start = time.perf_counter()

    result = None
    for _ in range(repeats):
        result = classical_bruteforce()

    elapsed = time.perf_counter() - start
    per_solve = elapsed / repeats

    return result, per_solve


# ============================================================
# 3. 메모리 측정 함수
# ============================================================

def measure_peak_rss(func, *args, sample_interval=0.002, **kwargs):
    """
    실행 전후의 process RSS를 반복적으로 확인하여
    peak RSS 증가량을 대략적으로 측정함

    주의:
    - 이 값은 PC process memory이며 실제 GPU Mepory가 아니고,
    - 실제 quantum hardware의 qubit memory도 아님
    """
    process = psutil.Process(os.getpid())
    baseline = process.memory_info().rss
    peak = baseline
    stop_event = threading.Event()

    def sampler():
        nonlocal peak

        while not stop_event.is_set():
            try:
                rss = process.memory_info().rss
                if rss > peak:
                    peak = rss
            except psutil.Error:
                pass

            time.sleep(sample_interval)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()

    start = time.perf_counter()

    try:
        result = func(*args, **kwargs)
    finally:
        elapsed = time.perf_counter() - start
        stop_event.set()
        thread.join()

    peak_increase_mb = max(0, peak - baseline) / (1024 ** 2)

    return result, elapsed, peak_increase_mb


# ============================================================
# 4. p=1 QAOA 회로 구성
# ============================================================

def build_qaoa_p1_circuit():
    """
    QAOA p=1 회로를 직접 구성함

    전체 흐름:
    H -> Cost Layer -> Mixer Layer -> Measurement

    1) H gate
       모든 비트열 후보를 superposition으로 준비함.

    2) Cost Layer
       각 edge마다 CX -> RZ(-gamma) -> CX를 적용함.
       이 부분이 Max-Cut cost 정보를 phase에 반영함.

    3) Mixer Layer
       모든 qubit에 RX(2*beta)를 적용함.
       amplitude를 섞어 측정 확률을 바꿈

    4) gamma, beta
       처음에는 알 수 없는 parameter이므로
       Classical Optimizer인 COBYLA로 찾음
    """
    gamma = Parameter("gamma")
    beta = Parameter("beta")

    qc = QuantumCircuit(N, N)

    # Step 1: 모든 후보를 superposition으로 준비
    qc.h(range(N))

    # Step 2: Cost Layer
    # 각 edge의 ZZ interaction을 CX-RZ-CX로 구현
    for i, j in EDGES:
        qc.cx(i, j)
        qc.rz(-gamma, j)
        qc.cx(i, j)

    # Step 3: Mixer Layer
    for q in range(N):
        qc.rx(2 * beta, q)

    # Step 4: 최종 측정
    qc.measure(range(N), range(N))

    return qc, gamma, beta


def qiskit_bitstring_to_x(bitstring):
    """
    Qiskit의 count bitstring은 c5...c0 순서로 표시됨
    이를 뒤집어서 x[0]이 q0에 대응하도록 맞춤
    """
    clean = bitstring.replace(" ", "")
    return tuple(int(b) for b in clean[::-1])


def metrics_from_counts(counts):
    """
    QAOA 측정 결과에서 세 가지 값을 계산함

    1) mean_cut
       전체 sample의 평균 Cut

    2) p_opt
       exact optimum Cut=5가 나온 비율

    3) best_sampled_cut
       측정된 sample 중 가장 좋은 Cut
    """
    total = sum(counts.values())

    mean_cut = 0.0
    optimal_count = 0
    best_sampled_cut = -1

    for bitstring, count in counts.items():
        x = qiskit_bitstring_to_x(bitstring)
        c = cut_value(x)

        mean_cut += c * count
        best_sampled_cut = max(best_sampled_cut, c)

        if c == 5:
            optimal_count += count

    mean_cut /= total
    p_opt = optimal_count / total

    return mean_cut, p_opt, best_sampled_cut


# ============================================================
# 5. QAOA 1회 optimization
# ============================================================

def qaoa_one_run(run_index, simulator, compiled_circuit, gamma, beta):
    """
    QAOA 1회 실행 과정

    (gamma, beta) 초기값 선택
        ->
    Quantum Circuit 실행
        ->
    Mean Cut 계산
        ->
    COBYLA가 gamma, beta 수정
        ->
    반복
        ->
    최종 parameter로 8192 shots 평가
    """
    rng = np.random.default_rng(BASE_SEED + run_index)

    # 각 QAOA run마다 다른 초기 angle 사용
    x0 = np.array([
        rng.uniform(0, 2 * np.pi),
        rng.uniform(0, np.pi),
    ])

    eval_counter = 0

    def objective(theta):
        """
        scipy.optimize.minimize는 값을 최소화함

        우리는 Mean Cut을 최대화하고 싶으므로
        objective = -Mean Cut으로 설정함
        """
        nonlocal eval_counter
        eval_counter += 1

        g, b = theta

        bound = compiled_circuit.assign_parameters(
            {gamma: g, beta: b},
            inplace=False,
        )

        seed_sim = int(rng.integers(1, 2_000_000_000))

        result = simulator.run(
            bound,
            shots=SHOTS_TRAIN,
            seed_simulator=seed_sim,
        ).result()

        counts = result.get_counts()
        mean_cut, _, _ = metrics_from_counts(counts)

        return -mean_cut

    # Classical optimizer가 QAOA angle을 반복적으로 찾음
    opt = minimize(
        objective,
        x0=x0,
        method="COBYLA",
        options={
            "maxiter": MAXITER,
            "rhobeg": 0.5,
            "tol": 1e-3,
        },
    )

    gamma_opt, beta_opt = opt.x

    # 최적화가 끝난 gamma, beta를 이용하여 최종 평가
    final_circuit = compiled_circuit.assign_parameters(
        {gamma: gamma_opt, beta: beta_opt},
        inplace=False,
    )

    final_seed = BASE_SEED + 100000 + run_index

    final_result = simulator.run(
        final_circuit,
        shots=SHOTS_FINAL,
        seed_simulator=final_seed,
    ).result()

    final_counts = final_result.get_counts()

    mean_cut, p_opt, best_sampled_cut = metrics_from_counts(final_counts)

    return {
        "run": run_index,
        "gamma": float(gamma_opt),
        "beta": float(beta_opt),
        "objective_evaluations": int(eval_counter),
        "mean_cut": float(mean_cut),
        "p_opt": float(p_opt),
        "best_sampled_cut": int(best_sampled_cut),
        "optimizer_success": bool(opt.success),
        "optimizer_message": str(opt.message),
    }


# ============================================================
# 6. Main Experiment
# ============================================================

def main():
    print("=" * 72)
    print("Homework #3: Classical Brute Force vs Gate-based QAOA")
    print("=" * 72)

    print("\n[Problem]")
    print("Nodes =", N)
    print("Edges =", EDGES)
    print("Possible bitstrings =", 2 ** N)

    # --------------------------------------------------------
    # A. Classical Brute Force
    # --------------------------------------------------------
    print("\n[1] Classical Brute Force")

    classical_result, classical_avg_time = benchmark_classical_time()
    best_q, best_solutions = classical_result

    # Classical memory는 1회 실행을 별도로 측정
    _, _, classical_peak_mb = measure_peak_rss(classical_bruteforce)

    classical_cut = -best_q

    print(f"Minimum QUBO value   = {best_q}")
    print(f"Maximum Cut          = {classical_cut}")
    print(f"# optimal bitstrings = {len(best_solutions)}")
    print(f"Average time/solve   = {classical_avg_time:.9f} s")
    print(f"Peak RSS increase    = {classical_peak_mb:.3f} MB")

    # --------------------------------------------------------
    # B. QAOA
    # --------------------------------------------------------
    print("\n[2] QAOA p=1 + COBYLA")
    print(f"Training shots/eval = {SHOTS_TRAIN}")
    print(f"Final shots/run     = {SHOTS_FINAL}")
    print(f"COBYLA maxiter      = {MAXITER}")
    print(f"Independent runs    = {N_QAOA_RUNS}")

    # 실제 quantum hardware가 아니라 AerSimulator 사용
    simulator = AerSimulator(method="statevector")

    qc, gamma, beta = build_qaoa_p1_circuit()

    # 같은 회로를 반복 사용하므로 transpile은 한 번만 수행
    compiled = transpile(qc, simulator, optimization_level=1)

    print("\nQAOA circuit:")
    print(qc.draw(output="text"))

    qaoa_rows = []

    for run_idx in range(1, N_QAOA_RUNS + 1):
        print(f"\n--- QAOA run {run_idx}/{N_QAOA_RUNS} ---")

        result, elapsed, peak_mb = measure_peak_rss(
            qaoa_one_run,
            run_idx,
            simulator,
            compiled,
            gamma,
            beta,
        )

        result["wall_time_s"] = float(elapsed)
        result["peak_rss_increase_mb"] = float(peak_mb)

        qaoa_rows.append(result)

        print(f"gamma*           = {result['gamma']:.6f}")
        print(f"beta*            = {result['beta']:.6f}")
        print(f"mean Cut         = {result['mean_cut']:.4f} / 5")
        print(f"P(opt)           = {100 * result['p_opt']:.2f}%")
        print(f"best sampled Cut = {result['best_sampled_cut']}")
        print(f"wall time        = {elapsed:.4f} s")
        print(f"peak RSS increase= {peak_mb:.3f} MB")

    # --------------------------------------------------------
    # C. QAOA 평균 결과
    # --------------------------------------------------------
    mean_cut_avg = np.mean([r["mean_cut"] for r in qaoa_rows])
    p_opt_avg = np.mean([r["p_opt"] for r in qaoa_rows])
    time_avg = np.mean([r["wall_time_s"] for r in qaoa_rows])
    mem_avg = np.mean([r["peak_rss_increase_mb"] for r in qaoa_rows])
    best_observed = max(r["best_sampled_cut"] for r in qaoa_rows)

    # --------------------------------------------------------
    # D. Final Comparison
    # --------------------------------------------------------
    print("\n" + "=" * 72)
    print("[3] Final comparison")
    print("=" * 72)

    print(f"{'Metric':<30} {'Classical':>18} {'QAOA':>18}")
    print("-" * 72)

    print(
        f"{'Best / exact Cut':<30} "
        f"{classical_cut:>18.4f} "
        f"{best_observed:>18.4f}"
    )

    print(
        f"{'Mean Cut':<30} "
        f"{'exact=5':>18} "
        f"{mean_cut_avg:>18.4f}"
    )

    print(
        f"{'P(opt)':<30} "
        f"{'exact search':>18} "
        f"{100 * p_opt_avg:>17.2f}%"
    )

    print(
        f"{'Wall time (s)':<30} "
        f"{classical_avg_time:>18.9f} "
        f"{time_avg:>18.4f}"
    )

    print(
        f"{'Peak RSS increase (MB)':<30} "
        f"{classical_peak_mb:>18.3f} "
        f"{mem_avg:>18.3f}"
    )

    # --------------------------------------------------------
    # E. Run-level CSV 저장
    # --------------------------------------------------------
    csv_path = "HW3_QAOA_Runs_LeeSeungwon_FINAL.csv"

    fieldnames = [
        "run",
        "gamma",
        "beta",
        "objective_evaluations",
        "mean_cut",
        "p_opt",
        "best_sampled_cut",
        "wall_time_s",
        "peak_rss_increase_mb",
        "optimizer_success",
        "optimizer_message",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(qaoa_rows)

    # --------------------------------------------------------
    # F. 결과 요약 TXT 저장
    # --------------------------------------------------------
    summary_path = "HW3_Result_Summary_LeeSeungwon_FINAL.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Homework #3 - Quantum Advantage for QUBO\n")
        f.write("Classical Brute Force vs Gate-based QAOA\n")
        f.write("=" * 60 + "\n\n")

        f.write("[1. Problem]\n")
        f.write("6-node Max-Cut QUBO\n")
        f.write(f"Edges = {EDGES}\n")
        f.write("Possible bitstrings = 64\n\n")

        f.write("[2. Classical Brute Force]\n")
        f.write(f"Minimum QUBO value = {best_q}\n")
        f.write(f"Maximum Cut = {classical_cut}\n")
        f.write(f"Optimal bitstrings = {len(best_solutions)}\n")
        f.write(
            f"Average execution time = "
            f"{classical_avg_time:.9f} s\n"
        )
        f.write(
            f"Peak RSS increase = "
            f"{classical_peak_mb:.3f} MB\n\n"
        )

        f.write("[3. QAOA]\n")
        f.write("p = 1\n")
        f.write("Optimizer = COBYLA\n")
        f.write(f"Training shots/evaluation = {SHOTS_TRAIN}\n")
        f.write(f"Final shots/run = {SHOTS_FINAL}\n")
        f.write(f"Independent runs = {N_QAOA_RUNS}\n")
        f.write(f"Mean Cut = {mean_cut_avg:.4f} / 5\n")
        f.write(f"Mean P(opt) = {100 * p_opt_avg:.2f}%\n")
        f.write(f"Best sampled Cut = {best_observed}\n")
        f.write(f"Average wall time = {time_avg:.4f} s\n")
        f.write(
            f"Average peak RSS increase = "
            f"{mem_avg:.3f} MB\n\n"
        )

        f.write("[4. Interpretation]\n")
        f.write(
            "Classical Brute Force checked all 64 possible bitstrings "
            "and obtained the exact solution Cut=5.\n"
        )
        f.write(
            "QAOA also sampled Cut=5, but the mean Cut and P(opt) "
            "show that the solution is probabilistic.\n"
        )
        f.write(
            "For this small problem, Classical Brute Force was much "
            "faster than QAOA simulation.\n"
        )
        f.write(
            "Therefore, Quantum Advantage was not observed in "
            "execution time or solution quality under this experiment.\n\n"
        )

        f.write("[5. Limitation]\n")
        f.write(
            "QAOA was executed with Qiskit AerSimulator, not real "
            "quantum hardware.\n"
        )
        f.write(
            "Therefore, simulator runtime should not be interpreted "
            "as quantum hardware speedup.\n"
        )
        f.write(
            "Peak RSS is host-process memory, not GPU memory or "
            "physical qubit memory.\n"
        )

    print("\nSaved:")
    print(" -", csv_path)
    print(" -", summary_path)

    print("\n[Interpretation caution]")
    print("QAOA uses Qiskit Aer simulation, not real quantum hardware.")
    print("Therefore, simulator time is not quantum hardware speedup.")
    print("Peak RSS is host-process memory, not GPU memory.")


if __name__ == "__main__":
    main()
