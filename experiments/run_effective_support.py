#!/usr/bin/env python3
"""Effective-support Markov prediction experiments.

This script uses the Python standard library for simulation and CSV output,
and matplotlib for JPG plots.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional, Sequence


PREDICTORS = ["MLE", "add-one", "add-half", "adaptive", "cesaro", "support-oracle"]


@dataclass
class Chain:
    K: int
    min_support: int
    max_support: int
    beta: float
    row_support_sizes: List[int]
    nexts: List[List[int]]
    probs: List[List[float]]
    pi: List[float]

    @property
    def avg_row_support(self) -> float:
        return sum(self.row_support_sizes) / len(self.row_support_sizes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-dir", default="results/effective_support_py")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--max-support", type=int, default=40)
    parser.add_argument("--betas", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--fixed-k", type=int, default=100)
    parser.add_argument("--fixed-n", type=int, default=1000)
    parser.add_argument("--fixed-beta", type=float, default=2.0)
    parser.add_argument("--k-grid", default="50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240")
    parser.add_argument("--n-grid", default="200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400,3600,3800,4000")
    return parser.parse_args()


def sample_from_prob(prob: Sequence[float]) -> int:
    u = random.random()
    total = 0.0
    for idx, p in enumerate(prob):
        total += p
        if u <= total:
            return idx
    return len(prob) - 1


def draw_stationary_distribution(nexts: Sequence[Sequence[int]], probs: Sequence[Sequence[float]], K: int, steps: int = 5000) -> List[float]:
    p = [1.0 / K] * K
    avg = [0.0] * K

    for _ in range(steps):
        p_next = [0.0] * K
        for i, support in enumerate(nexts):
            for j, prob in zip(support, probs[i]):
                p_next[j] += p[i] * prob
        p = p_next
        for i, value in enumerate(p):
            avg[i] += value

    total = sum(avg)
    pi = [0.0 if value / total < 1e-14 else value / total for value in avg]
    total = sum(pi)
    return [value / total for value in pi]


def power_law_probs(size: int, beta: float) -> List[float]:
    weights = [(rank + 1) ** (-beta) for rank in range(size)]
    total = sum(weights)
    return [weight / total for weight in weights]


def make_powerlaw_row_chain(K: int, min_support: int, max_support: int, beta: float) -> Chain:
    if min_support < 1:
        raise ValueError("min_support must be >= 1")
    if max_support > K:
        raise ValueError("max_support must be <= K")
    if min_support > max_support:
        raise ValueError("min_support must be <= max_support")

    row_support_sizes = [random.randint(min_support, max_support) for _ in range(K)]
    nexts = []
    probs = []
    for size in row_support_sizes:
        support = random.sample(range(K), size)
        random.shuffle(support)
        nexts.append(support)
        probs.append(power_law_probs(size, beta))
    pi = draw_stationary_distribution(nexts, probs, K)
    return Chain(
        K=K,
        min_support=min_support,
        max_support=max_support,
        beta=beta,
        row_support_sizes=row_support_sizes,
        nexts=nexts,
        probs=probs,
        pi=pi,
    )


def simulate_path(chain: Chain, n: int) -> List[int]:
    x = [sample_from_prob(chain.pi)]
    for _ in range(n - 1):
        row = x[-1]
        idx = sample_from_prob(chain.probs[row])
        x.append(chain.nexts[row][idx])
    return x


def terminal_counts(x: Sequence[int], K: int) -> List[int]:
    terminal = x[-1]
    counts = [0] * K
    for t in range(len(x) - 1):
        if x[t] == terminal:
            counts[x[t + 1]] += 1
    return counts


def kl_true(q: Sequence[float], true_support: Sequence[int], true_probs: Sequence[float]) -> float:
    tiny = float.fromhex("0x1.0p-1022")
    return sum(p * math.log(p / max(q[j], tiny)) for j, p in zip(true_support, true_probs))


def predict_losses(x: Sequence[int], chain: Chain, eps: float = 1e-12) -> Dict[str, float]:
    K = chain.K
    terminal = x[-1]
    true_support = chain.nexts[terminal]
    true_probs = chain.probs[terminal]
    counts = terminal_counts(x, K)
    row_n = sum(counts)
    distinct = sum(1 for count in counts if count > 0)

    if row_n == 0:
        q_mle = [1.0 / K] * K
    else:
        q_mle = [max(count / row_n, eps) for count in counts]
        total = sum(q_mle)
        q_mle = [value / total for value in q_mle]

    losses = {"MLE": kl_true(q_mle, true_support, true_probs)}

    for name, alpha in [("add-one", 1.0), ("add-half", 0.5)]:
        denom = row_n + K * alpha
        q = [(count + alpha) / denom for count in counts]
        losses[name] = kl_true(q, true_support, true_probs)

    if row_n == 0:
        q_adaptive = [1.0 / K] * K
    else:
        q_adaptive = [(count + distinct / K) / (row_n + distinct) for count in counts]
    losses["adaptive"] = kl_true(q_adaptive, true_support, true_probs)

    support_total = sum(counts[j] for j in true_support)
    q_support_oracle = [0.0] * K
    denom = support_total + 0.5 * len(true_support)
    for j in true_support:
        q_support_oracle[j] = (counts[j] + 0.5) / denom
    losses["support-oracle"] = kl_true(q_support_oracle, true_support, true_probs)

    losses["cesaro"] = cesaro_loss(x, chain, true_support, true_probs)
    return losses


def cesaro_loss(x: Sequence[int], chain: Chain, true_support: Sequence[int], true_probs: Sequence[float]) -> float:
    K = chain.K
    n = len(x)
    terminal = x[-1]
    counts = [0] * K
    row_n = 0
    distinct = 0
    q_sum = [0.0] * len(true_support)

    for start in range(n - 1, -1, -1):
        if start <= n - 2 and x[start] == terminal:
            nxt = x[start + 1]
            if counts[nxt] == 0:
                distinct += 1
            counts[nxt] += 1
            row_n += 1

        if row_n == 0:
            for idx in range(len(q_sum)):
                q_sum[idx] += 1.0 / K
        else:
            denom = row_n + distinct
            for idx, j in enumerate(true_support):
                q_sum[idx] += (counts[j] + distinct / K) / denom

    q = [value / n for value in q_sum]
    tiny = float.fromhex("0x1.0p-1022")
    return sum(p * math.log(p / max(q_value, tiny)) for p, q_value in zip(true_probs, q))


def effective_support(chain: Chain, n: int) -> float:
    total = 0.0
    for i, support in enumerate(chain.nexts):
        for prob in chain.probs[i]:
            total += min(n * chain.pi[i] * prob, 1.0)
    return total


def theory_rate(n: int, K: int, s: float) -> float:
    return (s / n) * (math.log(math.e * K * K / s) + math.log(math.e * n / s))


def parse_betas(raw: str) -> List[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def parse_ints(raw: str) -> List[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def scenario_grid(
    quick: bool,
    betas: Sequence[float],
    fixed_k: int,
    fixed_n: int,
    fixed_beta: float,
    k_grid: Sequence[int],
    n_grid: Sequence[int],
) -> List[Dict[str, object]]:
    if quick:
        k_grid = [80, 100, 120]
        n_grid = [400, 800, 1200]
        fixed_k = 100
        fixed_n = 800
        fixed_beta = betas[min(len(betas) - 1, 1)]

    scenarios = []
    scenarios.extend({"K": fixed_k, "n": fixed_n, "beta": beta} for beta in betas)
    scenarios.extend({"K": fixed_k, "n": n, "beta": fixed_beta} for n in n_grid)
    scenarios.extend({"K": k, "n": fixed_n, "beta": fixed_beta} for k in k_grid)

    unique = {}
    for scenario in scenarios:
        unique[(int(scenario["K"]), int(scenario["n"]), float(scenario["beta"]))] = scenario
    return list(unique.values())


def summarise_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[tuple, List[Dict[str, object]]] = defaultdict(list)
    key_cols = ("K", "n", "min_support", "max_support", "beta", "avg_row_support", "s_eff", "rate", "predictor")
    for row in rows:
        groups[tuple(row[col] for col in key_cols)].append(row)

    summary = []
    for key, group in groups.items():
        losses = [float(row["loss"]) for row in group]
        mean_loss = sum(losses) / len(losses)
        if len(losses) > 1:
            var = sum((loss - mean_loss) ** 2 for loss in losses) / (len(losses) - 1)
            se_loss = math.sqrt(var) / math.sqrt(len(losses))
        else:
            se_loss = 0.0
        out = {col: value for col, value in zip(key_cols, key)}
        out.update(
            mean_loss=mean_loss,
            se_loss=se_loss,
            median_loss=median(losses),
            mean_ratio_to_rate=mean_loss / float(out["rate"]),
            reps=len(losses),
        )
        summary.append(out)
    summary.sort(key=lambda row: (float(row["rate"]), str(row["predictor"])))
    return summary


def write_csv(path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str) -> List[Dict[str, object]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def choose_value(values: Sequence[float], preferred: float) -> float:
    return min(values, key=lambda value: abs(value - preferred))


def fitted_theory_constant(rows: Sequence[Dict[str, object]]) -> Optional[float]:
    cesaro_rows = [
        row
        for row in rows
        if row["predictor"] == "cesaro" and float(row["mean_loss"]) > 0 and float(row["rate"]) > 0
    ]
    if not cesaro_rows:
        return None
    log_c = sum(math.log(float(row["mean_loss"]) / float(row["rate"])) for row in cesaro_rows) / len(cesaro_rows)
    return math.exp(log_c)


def add_smooth_theory_curve(plt: object, rows: Sequence[Dict[str, object]], grid_size: int = 200) -> None:
    fitted_c = fitted_theory_constant(rows)
    if fitted_c is None:
        return
    k_values = {float(row["K"]) for row in rows}
    n_values = {float(row["n"]) for row in rows}
    if len(k_values) != 1 or len(n_values) != 1:
        return

    K = int(next(iter(k_values)))
    n = int(next(iter(n_values)))
    s_values = [float(row["s_eff"]) for row in rows if float(row["s_eff"]) > 0]
    if len(s_values) < 2:
        return
    s_min = min(s_values)
    s_max = max(s_values)
    if s_min >= s_max:
        return

    log_min = math.log(s_min)
    log_max = math.log(s_max)
    xs = [math.exp(log_min + idx * (log_max - log_min) / (grid_size - 1)) for idx in range(grid_size)]
    ys = [fitted_c * theory_rate(n, K, s) for s in xs]
    plt.plot(xs, ys, linestyle=":", linewidth=2.5, color="black", label=f"theory form, C={fitted_c:.2g}")


def plot_risk_vs_support_slice(
    plt: object,
    summary: Sequence[Dict[str, object]],
    predictors: Sequence[str],
    colors: Dict[str, str],
    fixed: Dict[str, float],
    varying: str,
    title: str,
    path: str,
    draw_smooth_theory: bool = False,
) -> None:
    rows = [
        row
        for row in summary
        if all(abs(float(row[key]) - value) < 1e-12 for key, value in fixed.items())
    ]
    if not rows:
        return

    plt.figure(figsize=(9, 6))
    for predictor in predictors:
        predictor_rows = sorted(
            [row for row in rows if row["predictor"] == predictor],
            key=lambda row: float(row["s_eff"]),
        )
        if not predictor_rows:
            continue
        xs = [float(row["s_eff"]) for row in predictor_rows]
        ys = [max(float(row["mean_loss"]), 1e-12) for row in predictor_rows]
        plt.scatter(xs, ys, s=36, label=predictor, color=colors[predictor])

    if draw_smooth_theory:
        add_smooth_theory_curve(plt, rows)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("effective support size s_n(mu)")
    plt.ylabel("empirical next-row KL risk")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_plots(summary: Sequence[Dict[str, object]], out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib is required because this script is configured to write JPG plots only")

    predictors = list(dict.fromkeys(row["predictor"] for row in summary))
    colors = {
        predictor: color
        for predictor, color in zip(predictors, ["#1b6ca8", "#d1495b", "#edae49", "#00798c", "#7a5195", "#4d908e"])
    }

    plt.figure(figsize=(9, 6))
    for predictor in predictors:
        rows = [row for row in summary if row["predictor"] == predictor]
        plt.scatter(
            [float(row["s_eff"]) for row in rows],
            [max(float(row["mean_loss"]), 1e-12) for row in rows],
            label=predictor,
            color=colors[predictor],
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("effective support size s_n(mu)")
    plt.ylabel("empirical next-row KL risk")
    plt.title("Empirical risk vs effective support size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "risk_vs_effective_support.jpg"), dpi=180)
    plt.close()

    medians = []
    for predictor in predictors:
        values = [float(row["mean_loss"]) for row in summary if row["predictor"] == predictor]
        medians.append((predictor, median(values)))
    medians.sort(key=lambda item: item[1])
    plt.figure(figsize=(11, 7))
    plt.bar([item[0] for item in medians], [item[1] for item in medians], color=[colors[item[0]] for item in medians])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("median across-scenario empirical KL risk")
    plt.title("Predictor comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "predictor_comparison.jpg"), dpi=180)
    plt.close()

    if summary and "beta" in summary[0]:
        betas = sorted({float(row["beta"]) for row in summary})
        s_values = []
        for beta in betas:
            beta_s = [float(row["s_eff"]) for row in summary if abs(float(row["beta"]) - beta) < 1e-12]
            s_values.append(median(beta_s))
        plt.figure(figsize=(9, 6))
        plt.plot(betas, s_values, marker="o", linewidth=2, color="#1b6ca8")
        plt.xlabel("power-law exponent beta")
        plt.ylabel("median effective support size s_n(mu)")
        plt.title("Effective support size across beta")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "effective_support_by_beta.jpg"), dpi=180)
        plt.close()

        plt.figure(figsize=(10, 6))
        for predictor in predictors:
            values = []
            for beta in betas:
                beta_rows = [
                    float(row["mean_loss"])
                    for row in summary
                    if row["predictor"] == predictor and abs(float(row["beta"]) - beta) < 1e-12
                ]
                values.append(median(beta_rows))
            plt.plot(betas, values, marker="o", linewidth=2, label=predictor, color=colors[predictor])
        plt.xlabel("power-law exponent beta")
        plt.ylabel("median empirical KL risk")
        plt.title("Predictor comparison across beta")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "predictor_comparison_by_beta.jpg"), dpi=180)
        plt.close()

        ks = sorted({float(row["K"]) for row in summary})
        ns = sorted({float(row["n"]) for row in summary})
        fixed_k = choose_value(ks, 100)
        fixed_n = choose_value(ns, 1000)
        fixed_beta = choose_value(betas, 2)

        plot_risk_vs_support_slice(
            plt,
            summary,
            predictors,
            colors,
            fixed={"K": fixed_k, "n": fixed_n},
            varying="beta",
            title=f"Risk vs effective support, varying beta (K={fixed_k:g}, n={fixed_n:g})",
            path=os.path.join(out_dir, "risk_vs_support_vary_beta_fixed_K_n.jpg"),
            draw_smooth_theory=True,
        )
        plot_risk_vs_support_slice(
            plt,
            summary,
            predictors,
            colors,
            fixed={"K": fixed_k, "beta": fixed_beta},
            varying="n",
            title=f"Risk vs effective support, varying n (K={fixed_k:g}, beta={fixed_beta:g})",
            path=os.path.join(out_dir, "risk_vs_support_vary_n_fixed_K_beta.jpg"),
        )
        plot_risk_vs_support_slice(
            plt,
            summary,
            predictors,
            colors,
            fixed={"n": fixed_n, "beta": fixed_beta},
            varying="K",
            title=f"Risk vs effective support, varying K (n={fixed_n:g}, beta={fixed_beta:g})",
            path=os.path.join(out_dir, "risk_vs_support_vary_K_fixed_n_beta.jpg"),
        )


def print_predictor_summary(summary: Sequence[Dict[str, object]]) -> None:
    by_predictor = defaultdict(list)
    for row in summary:
        by_predictor[row["predictor"]].append(float(row["mean_loss"]))
    for predictor, values in sorted(by_predictor.items(), key=lambda item: median(item[1])):
        print(f"{predictor:16s} {median(values):.7g}")


def main() -> None:
    args = parse_args()
    if args.quick:
        args.reps = min(args.reps, 20)
    betas = parse_betas(args.betas)
    k_grid = parse_ints(args.k_grid)
    n_grid = parse_ints(args.n_grid)
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    raw_path = os.path.join(args.out_dir, "raw_losses.csv")
    summary_path = os.path.join(args.out_dir, "summary.csv")

    if args.plot_only:
        summary = read_csv(summary_path)
        write_plots(summary, args.out_dir)
        print(f"rewrote plots in {os.path.abspath(args.out_dir)}")
        return

    grid = scenario_grid(args.quick, betas, args.fixed_k, args.fixed_n, args.fixed_beta, k_grid, n_grid)
    total = len(grid) * args.reps
    done = 0
    rows = []

    for scenario in grid:
        K = scenario["K"]
        n = scenario["n"]
        beta = float(scenario["beta"])
        chain = make_powerlaw_row_chain(K, args.min_support, args.max_support, beta)
        s_eff = effective_support(chain, n)
        rate = theory_rate(n, K, s_eff)

        for rep in range(1, args.reps + 1):
            x = simulate_path(chain, n)
            losses = predict_losses(x, chain)
            for predictor in PREDICTORS:
                rows.append(
                    {
                        "K": K,
                        "n": n,
                        "min_support": args.min_support,
                        "max_support": args.max_support,
                        "beta": beta,
                        "avg_row_support": chain.avg_row_support,
                        "s_eff": s_eff,
                        "rate": rate,
                        "rep": rep,
                        "predictor": predictor,
                        "loss": losses[predictor],
                    }
                )
            done += 1
            if done % max(1, total // 20) == 0:
                print(f"completed {done}/{total} simulated trajectories")

    summary = summarise_rows(rows)
    write_csv(
        raw_path,
        rows,
        ["K", "n", "min_support", "max_support", "beta", "avg_row_support", "s_eff", "rate", "rep", "predictor", "loss"],
    )
    write_csv(
        summary_path,
        summary,
        [
            "K",
            "n",
            "min_support",
            "max_support",
            "beta",
            "avg_row_support",
            "s_eff",
            "rate",
            "predictor",
            "mean_loss",
            "se_loss",
            "median_loss",
            "mean_ratio_to_rate",
            "reps",
        ],
    )
    write_plots(summary, args.out_dir)
    print(f"wrote {os.path.abspath(raw_path)}")
    print(f"wrote {os.path.abspath(summary_path)}")
    print_predictor_summary(summary)


if __name__ == "__main__":
    main()
