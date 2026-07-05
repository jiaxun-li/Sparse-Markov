# Effective-support Markov prediction experiments

This folder contains the Python simulation for sparse Markov prediction experiments.

## Power-Law Row Model

`run_effective_support.py` simulates stationary sparse Markov chains on a nominal alphabet of size `K`.
For each state `i`, the row support size is drawn uniformly as

```text
s_i ~ Uniform{min_support, ..., max_support}.
```

Then `s_i` available next states are sampled without replacement from `{1, ..., K}`. Conditional on
that support, the true transition probabilities follow a rank power law:

```text
M(j_r | i) proportional to r^{-beta},   r = 1, ..., s_i.
```

When `beta = 0`, each row is uniform on its support. Larger `beta` makes each row more concentrated:
the top-ranked successors get most of the mass while low-ranked successors become rare.

The script computes a stationary distribution for the generated chain and evaluates the stationary
edge effective support

```text
s_n(mu) = sum_i sum_j min{ n pi_i M_ij, 1 }.
```

For each simulated trajectory, it evaluates terminal-row prediction risk

```text
KL( M(. | X_n) || M_hat(. | X_n) )
```

and compares it with the effective-support rate

```text
(s / n) * { log(e K^2 / s) + log(e n / s) }.
```

## Predictors

- `MLE`: row MLE with tiny probability clipping to avoid infinite KL.
- `add-one`: full-alphabet add-one smoothing.
- `add-half`: full-alphabet add-half smoothing.
- `adaptive`: row-wise adaptive smoothing from the note.
- `cesaro`: Cesaro average of adaptive suffix predictors.
- `support-oracle`: add-half smoothing restricted to the true active successors.

## Run

On this machine, `D:\Anaconda\envs\d2l-zh\python.exe` has a working matplotlib install and writes JPG plots:

```powershell
& 'D:\Anaconda\envs\d2l-zh\python.exe' '.\experiments\run_effective_support.py' --reps=100 --seed=11 --min-support=2 --max-support=40 --betas=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 --out-dir='results/powerlaw_row_beta'
```

The default full run uses a targeted slice grid:

- fixed `K = 100`, `n = 1000`, varying `beta`;
- fixed `K = 100`, `beta = 2`, varying 20 values of `n`;
- fixed `n = 1000`, `beta = 2`, varying 20 values of `K`.

Quick smoke test:

```powershell
& 'D:\Anaconda\envs\d2l-zh\python.exe' '.\experiments\run_effective_support.py' --quick --reps=5 --seed=7 --min-support=2 --max-support=10 --betas=0,1,2 --out-dir='results/powerlaw_row_quick'
```

Regenerate only the JPG plots from an existing `summary.csv`:

```powershell
& 'D:\Anaconda\envs\d2l-zh\python.exe' '.\experiments\run_effective_support.py' --plot-only --out-dir='results/powerlaw_row_beta'
```

Fixed-chain sweep over `n = 1, ..., 50000`, comparing all predictors with one fixed generated `M` and `pi`; the plot includes a smooth fitted theory curve `C * rate(n, K, s_n(mu))`:

```powershell
& 'D:\Anaconda\envs\d2l-zh\python.exe' '.\experiments\run_effective_support.py' --fixed-chain-n-sweep --reps=50 --seed=17 --fixed-k=100 --fixed-beta=2 --min-support=2 --max-support=40 --n-sweep-max=50000 --n-sweep-count=120 --out-dir='results/powerlaw_row_beta'
```

Grid sweep over 10 values of `K` from 100 to 1000, 10 values of `beta` from 1 to 10, and 1000 values of `n` from 10 to 10000; the plot uses the theory rate as the x-axis and does not draw a fitted reference line. This large grid omits exact `cesaro` because computing the suffix average at all 100000 grid points is much slower than the other predictors:

```powershell
& 'D:\Anaconda\envs\d2l-zh\python.exe' '.\experiments\run_effective_support.py' --grid-rate-plot --reps=1 --seed=23 --min-support=2 --max-support=40 --grid-k-count=10 --grid-beta-count=10 --grid-n-count=1000 --grid-stationary-steps=300 --out-dir='results/powerlaw_row_beta'
```

## Outputs

- `raw_losses.csv`: one row per trajectory, scenario, and predictor.
- `summary.csv`: mean/median KL risk, standard error, and risk/rate ratio.
- `risk_vs_effective_support.jpg`: empirical risk against the effective support size `s_n(mu)`.
- `predictor_comparison.jpg`: median across-scenario risk by predictor.
- `effective_support_by_beta.jpg`: median effective support size as `beta` varies.
- `predictor_comparison_by_beta.jpg`: median empirical KL risk for each predictor as `beta` varies.
- `risk_vs_support_vary_beta_fixed_K_n.jpg`: fixed `K,n`, varying `beta`, with a smooth fitted theory curve as a function of `s_n(mu)`.
- `risk_vs_support_vary_n_fixed_K_beta.jpg`: fixed `K,beta`, varying `n`.
- `risk_vs_support_vary_K_fixed_n_beta.jpg`: fixed `n,beta`, varying `K`.
- `fixed_chain_predictors_by_n.jpg`: fixed generated chain, varying `n`, comparing all predictors, with a smooth fitted theory curve.
- `fixed_chain_risk_vs_theory_rate.jpg`: same fixed-chain sweep, with x-axis equal to the theory rate and y-axis equal to empirical next-row KL risk.
- `fixed_chain_n_sweep_summary.csv`: summary table for the fixed-chain `n` sweep.
- `grid_risk_vs_theory_rate.jpg`: grid sweep across `K`, `beta`, and `n`, with empirical risk plotted against the theory rate.
- `grid_rate_summary.csv`: summary table for the grid sweep.
