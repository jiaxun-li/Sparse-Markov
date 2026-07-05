#!/usr/bin/env python3
"""Effective-support Markov prediction experiments.

This script uses the Python standard library for simulation and CSV output,
and matplotlib for JPG plots.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional, Sequence


PREDICTORS = ["MLE", "add-one", "add-half", "adaptive", "cesaro", "support-oracle"]
GRID_PREDICTORS = ["MLE", "add-one", "add-half", "adaptive", "support-oracle"]


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
    parser.add_argument("--fixed-chain-n-sweep", action="store_true")
    parser.add_argument("--n-sweep-max", type=int, default=50000)
    parser.add_argument("--n-sweep-count", type=int, default=120)
    parser.add_argument("--grid-rate-plot", action="store_true")
    parser.add_argument("--grid-k-count", type=int, default=10)
    parser.add_argument("--grid-beta-count", type=int, default=10)
    parser.add_argument("--grid-n-count", type=int, default=1000)
    parser.add_argument("--grid-stationary-steps", type=int, default=300)
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


def make_powerlaw_row_chain(K: int, min_support: int, max_support: int, beta: float, stationary_steps: int = 5000) -> Chain:
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
    pi = draw_stationary_distribution(nexts, probs, K, steps=stationary_steps)
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


def index_row_events(path: Sequence[int], K: int) -> List[List[tuple]]:
    events = [[] for _ in range(K)]
    for t in range(len(path) - 1):
        events[path[t]].append((t, path[t + 1]))
    return events


def row_event_counts(events: Sequence[tuple], cutoff_time: int) -> tuple:
    times = [event[0] for event in events]
    cutoff = bisect.bisect_left(times, cutoff_time)
    counts = defaultdict(int)
    for _, nxt in events[:cutoff]:
        counts[nxt] += 1
    return counts, cutoff


def cesaro_loss_from_events(
    events: Sequence[tuple],
    n: int,
    K: int,
    true_support: Sequence[int],
    true_probs: Sequence[float],
) -> float:
    relevant = [event for event in events if event[0] < n - 1]
    counts = defaultdict(int)
    distinct = 0
    row_n = 0
    q_sum = [0.0] * len(true_support)
    previous_start = n

    for t, nxt in reversed(relevant):
        segment_length = previous_start - (t + 1)
        if segment_length > 0:
            if row_n == 0:
                for idx in range(len(q_sum)):
                    q_sum[idx] += segment_length / K
            else:
                denom = row_n + distinct
                for idx, j in enumerate(true_support):
                    q_sum[idx] += segment_length * ((counts[j] + distinct / K) / denom)

        if counts[nxt] == 0:
            distinct += 1
        counts[nxt] += 1
        row_n += 1

        denom = row_n + distinct
        for idx, j in enumerate(true_support):
            q_sum[idx] += (counts[j] + distinct / K) / denom
        previous_start = t

    if previous_start > 0:
        segment_length = previous_start
        if row_n == 0:
            for idx in range(len(q_sum)):
                q_sum[idx] += segment_length / K
        else:
            denom = row_n + distinct
            for idx, j in enumerate(true_support):
                q_sum[idx] += segment_length * ((counts[j] + distinct / K) / denom)

    q = [value / n for value in q_sum]
    tiny = float.fromhex("0x1.0p-1022")
    return sum(p * math.log(p / max(q_value, tiny)) for p, q_value in zip(true_probs, q))


def predict_losses_from_events(
    n: int,
    path: Sequence[int],
    chain: Chain,
    row_events: Sequence[Sequence[tuple]],
    eps: float = 1e-12,
    include_cesaro: bool = True,
) -> Dict[str, float]:
    K = chain.K
    terminal = path[n - 1]
    true_support = chain.nexts[terminal]
    true_probs = chain.probs[terminal]
    counts, row_n = row_event_counts(row_events[terminal], n - 1)
    distinct = len(counts)

    if row_n == 0:
        q_mle_support = {j: 1.0 / K for j in true_support}
    else:
        q_mle = {j: max(counts[j] / row_n, eps) for j in true_support}
        clipped_unseen = K - len(counts)
        total = sum(max(count / row_n, eps) for count in counts.values()) + clipped_unseen * eps
        q_mle_support = {j: q_mle[j] / total for j in true_support}
    losses = {"MLE": sum(p * math.log(p / max(q_mle_support[j], float.fromhex("0x1.0p-1022"))) for j, p in zip(true_support, true_probs))}

    for name, alpha in [("add-one", 1.0), ("add-half", 0.5)]:
        denom = row_n + K * alpha
        q = [(counts[j] + alpha) / denom for j in true_support]
        losses[name] = sum(p * math.log(p / q_value) for p, q_value in zip(true_probs, q))

    if row_n == 0:
        q_adaptive = [1.0 / K] * len(true_support)
    else:
        q_adaptive = [(counts[j] + distinct / K) / (row_n + distinct) for j in true_support]
    losses["adaptive"] = sum(p * math.log(p / max(q_value, float.fromhex("0x1.0p-1022"))) for p, q_value in zip(true_probs, q_adaptive))

    support_total = sum(counts[j] for j in true_support)
    denom = support_total + 0.5 * len(true_support)
    q_support_oracle = [(counts[j] + 0.5) / denom for j in true_support]
    losses["support-oracle"] = sum(p * math.log(p / q_value) for p, q_value in zip(true_probs, q_support_oracle))
    if include_cesaro:
        losses["cesaro"] = cesaro_loss_from_events(row_events[terminal], n, K, true_support, true_probs)
    return losses


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


def log_spaced_ints(start: int, stop: int, count: int) -> List[int]:
    if start < 1 or stop < start:
        raise ValueError("need 1 <= start <= stop")
    if count <= 1:
        return [start, stop] if start != stop else [start]

    values = {start, stop}
    log_start = math.log(start)
    log_stop = math.log(stop)
    for idx in range(count):
        value = round(math.exp(log_start + idx * (log_stop - log_start) / (count - 1)))
        values.add(max(start, min(stop, int(value))))
    values.update(range(start, min(stop, 20) + 1))
    power = 1
    while power <= stop:
        if power >= start:
            values.add(power)
        power *= 10
    return sorted(values)


def linearly_spaced_ints(start: int, stop: int, count: int) -> List[int]:
    if start < 1 or stop < start:
        raise ValueError("need 1 <= start <= stop")
    if count <= 1:
        return [start]
    values = [round(start + idx * (stop - start) / (count - 1)) for idx in range(count)]
    return sorted(set(int(value) for value in values))


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


def add_fixed_chain_n_theory_curve(
    plt: object,
    chain: Chain,
    summary: Sequence[Dict[str, object]],
    n_min: int,
    n_max: int,
    grid_size: int = 300,
) -> None:
    fitted_c = fitted_theory_constant(summary)
    if fitted_c is None:
        return
    log_min = math.log(n_min)
    log_max = math.log(n_max)
    xs = [math.exp(log_min + idx * (log_max - log_min) / (grid_size - 1)) for idx in range(grid_size)]
    ys = []
    for n_value in xs:
        s_value = effective_support(chain, n_value)
        ys.append(fitted_c * theory_rate(n_value, chain.K, s_value))
    plt.plot(xs, ys, linestyle=":", linewidth=2.5, color="black", label=f"theory form, C={fitted_c:.2g}")


def predictor_colors(predictors: Sequence[str]) -> Dict[str, str]:
    palette = ["#0072B2", "#E69F00", "#CC79A7", "#D55E00", "#009E73", "#56B4E9"]
    return {predictor: color for predictor, color in zip(predictors, palette)}


def write_fixed_chain_plots(summary: Sequence[Dict[str, object]], chain: Chain, n_values: Sequence[int], out_dir: str) -> List[str]:
    import matplotlib.pyplot as plt

    colors = predictor_colors(PREDICTORS)
    plot_paths = []

    plt.figure(figsize=(10, 6))
    for predictor in PREDICTORS:
        rows = sorted([row for row in summary if row["predictor"] == predictor], key=lambda row: int(row["n"]))
        plt.plot(
            [int(row["n"]) for row in rows],
            [max(float(row["mean_loss"]), 1e-12) for row in rows],
            marker="o",
            markersize=3,
            linewidth=1.6,
            label=predictor,
            color=colors[predictor],
        )
    add_fixed_chain_n_theory_curve(plt, chain, summary, min(n_values), max(n_values))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("sample length n")
    plt.ylabel("empirical next-row KL risk")
    plt.title(f"Predictor comparison with fixed chain (K={chain.K}, beta={chain.beta:g})")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "fixed_chain_predictors_by_n.jpg")
    plt.savefig(plot_path, dpi=180)
    plt.close()
    plot_paths.append(plot_path)

    plt.figure(figsize=(10, 6))
    for predictor in PREDICTORS:
        rows = sorted([row for row in summary if row["predictor"] == predictor], key=lambda row: float(row["rate"]))
        plt.plot(
            [float(row["rate"]) for row in rows],
            [max(float(row["mean_loss"]), 1e-12) for row in rows],
            marker="o",
            markersize=3,
            linewidth=1.6,
            label=predictor,
            color=colors[predictor],
        )
    fitted_c = fitted_theory_constant(summary)
    if fitted_c is not None:
        rates = [float(row["rate"]) for row in summary if float(row["rate"]) > 0]
        if rates:
            x_min = min(rates)
            x_max = max(rates)
            log_min = math.log(x_min)
            log_max = math.log(x_max)
            xs = [math.exp(log_min + idx * (log_max - log_min) / 299) for idx in range(300)]
            ys = [fitted_c * x for x in xs]
            plt.plot(xs, ys, linestyle=":", linewidth=2.5, color="black", label=f"theory form, C={fitted_c:.2g}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("theory rate")
    plt.ylabel("empirical next-row KL risk")
    plt.title(f"Empirical risk vs theory rate with fixed chain (K={chain.K}, beta={chain.beta:g})")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "fixed_chain_risk_vs_theory_rate.jpg")
    plt.savefig(plot_path, dpi=180)
    plt.close()
    plot_paths.append(plot_path)

    return plot_paths


def write_grid_rate_plot(summary: Sequence[Dict[str, object]], out_dir: str) -> str:
    import matplotlib.pyplot as plt

    predictors = list(dict.fromkeys(row["predictor"] for row in summary))
    colors = predictor_colors(predictors)
    markers = ["o", "s", "^", "D", "P", "X"]

    plt.figure(figsize=(11, 6.5))
    for predictor in predictors:
        rows = [row for row in summary if row["predictor"] == predictor]
        marker = markers[predictors.index(predictor) % len(markers)]
        plt.scatter(
            [float(row["rate"]) for row in rows],
            [max(float(row["mean_loss"]), 1e-12) for row in rows],
            s=8,
            alpha=0.7,
            label=predictor,
            color=colors[predictor],
            marker=marker,
            edgecolors="none",
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("theory rate")
    plt.ylabel("empirical next-row KL risk")
    plt.title("Empirical risk vs theory rate across K, beta, and n")
    plt.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.45)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0, markerscale=2.5)
    plt.tight_layout()
    path = os.path.join(out_dir, "grid_risk_vs_theory_rate.jpg")
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def write_grid_rate_experiment(args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        raise RuntimeError("matplotlib is required because this script is configured to write JPG plots only")

    k_values = linearly_spaced_ints(100, 1000, args.grid_k_count)
    beta_values = linearly_spaced_ints(1, 10, args.grid_beta_count)
    n_values = linearly_spaced_ints(10, 10000, args.grid_n_count)
    raw_rows = []
    total = len(k_values) * len(beta_values) * args.reps
    done = 0

    for K in k_values:
        for beta in beta_values:
            chain = make_powerlaw_row_chain(
                K,
                args.min_support,
                min(args.max_support, K),
                float(beta),
                stationary_steps=args.grid_stationary_steps,
            )
            s_by_n = {n: effective_support(chain, n) for n in n_values}
            rate_by_n = {n: theory_rate(n, K, s_by_n[n]) for n in n_values}
            for rep in range(1, args.reps + 1):
                path = simulate_path(chain, max(n_values))
                row_events = index_row_events(path, K)
                for n in n_values:
                    losses = predict_losses_from_events(n, path, chain, row_events, include_cesaro=False)
                    for predictor in GRID_PREDICTORS:
                        raw_rows.append(
                            {
                                "K": K,
                                "n": n,
                                "min_support": args.min_support,
                                "max_support": min(args.max_support, K),
                                "beta": float(beta),
                                "avg_row_support": chain.avg_row_support,
                                "s_eff": s_by_n[n],
                                "rate": rate_by_n[n],
                                "rep": rep,
                                "predictor": predictor,
                                "loss": losses[predictor],
                            }
                        )
                done += 1
                if done % max(1, total // 100) == 0:
                    print(f"completed {done}/{total} grid trajectories")

    summary = summarise_rows(raw_rows)
    raw_path = os.path.join(args.out_dir, "grid_rate_raw_losses.csv")
    summary_path = os.path.join(args.out_dir, "grid_rate_summary.csv")
    write_csv(
        raw_path,
        raw_rows,
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
    plot_path = write_grid_rate_plot(summary, args.out_dir)
    print(f"wrote {os.path.abspath(raw_path)}")
    print(f"wrote {os.path.abspath(summary_path)}")
    print(f"wrote {os.path.abspath(plot_path)}")


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


def write_fixed_chain_n_sweep(args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib is required because this script is configured to write JPG plots only")

    chain = make_powerlaw_row_chain(args.fixed_k, args.min_support, args.max_support, args.fixed_beta)
    n_values = log_spaced_ints(1, args.n_sweep_max, args.n_sweep_count)
    raw_rows = []
    s_by_n = {n: effective_support(chain, n) for n in n_values}
    rate_by_n = {n: theory_rate(n, chain.K, s_by_n[n]) for n in n_values}

    for rep in range(1, args.reps + 1):
        path = simulate_path(chain, args.n_sweep_max)
        for n in n_values:
            losses = predict_losses(path[:n], chain)
            for predictor in PREDICTORS:
                raw_rows.append(
                    {
                        "K": chain.K,
                        "n": n,
                        "min_support": chain.min_support,
                        "max_support": chain.max_support,
                        "beta": chain.beta,
                        "avg_row_support": chain.avg_row_support,
                        "s_eff": s_by_n[n],
                        "rate": rate_by_n[n],
                        "rep": rep,
                        "predictor": predictor,
                        "loss": losses[predictor],
                    }
                )
        if rep % max(1, args.reps // 10) == 0:
            print(f"completed {rep}/{args.reps} fixed-chain n-sweep trajectories")

    summary = summarise_rows(raw_rows)
    raw_path = os.path.join(args.out_dir, "fixed_chain_n_sweep_raw_losses.csv")
    summary_path = os.path.join(args.out_dir, "fixed_chain_n_sweep_summary.csv")
    write_csv(
        raw_path,
        raw_rows,
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

    plot_paths = write_fixed_chain_plots(summary, chain, n_values, args.out_dir)

    print(f"wrote {os.path.abspath(raw_path)}")
    print(f"wrote {os.path.abspath(summary_path)}")
    for plot_path in plot_paths:
        print(f"wrote {os.path.abspath(plot_path)}")


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

    if args.grid_rate_plot:
        write_grid_rate_experiment(args)
        return

    if args.fixed_chain_n_sweep:
        write_fixed_chain_n_sweep(args)
        return

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
