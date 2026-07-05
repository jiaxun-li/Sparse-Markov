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
- `oracle`: add-half smoothing restricted to the true active successors.

## Run

On this machine, `D:\Anaconda\envs\d2l-zh\python.exe` has a working matplotlib install and writes JPG plots:

```powershell
& 'D:\Anaconda\envs\d2l-zh\python.exe' '.\experiments\run_effective_support.py' --reps=100 --seed=11 --min-support=2 --max-support=40 --betas=0,0.5,1,1.5,2 --out-dir='results/powerlaw_row'
```

Quick smoke test:

```powershell
& 'D:\Anaconda\envs\d2l-zh\python.exe' '.\experiments\run_effective_support.py' --quick --reps=5 --seed=7 --min-support=2 --max-support=10 --betas=0,1,2 --out-dir='results/powerlaw_row_quick'
```

Regenerate only the JPG plots from an existing `summary.csv`:

```powershell
& 'D:\Anaconda\envs\d2l-zh\python.exe' '.\experiments\run_effective_support.py' --plot-only --out-dir='results/powerlaw_row'
```

## Outputs

- `raw_losses.csv`: one row per trajectory, scenario, and predictor.
- `summary.csv`: mean/median KL risk, standard error, and risk/rate ratio.
- `risk_vs_theory_rate.jpg`: empirical risk against the effective-support rate.
- `predictor_comparison.jpg`: median across-scenario risk by predictor.
- `predictor_comparison_by_beta.jpg`: median empirical KL risk for each predictor as `beta` varies.
