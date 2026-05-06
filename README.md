# Kernel-GCA (Paper-style Results)

This package extends the Kernel-GCA implementation with a paper-style experiment that produces:
- Connected VAR-like chain (s1 -> s2 -> s3) vs independent null control
- Objective-at-convergence curves across recovered pairs
- Predictive gain (Granger-like) metrics on train and test
- Baselines: PCA and lightweight FastICA
- Fidelity table (LaTeX) and representative scatter plots

## Run
```bash
pip install -r requirements.txt
```

## Create datasets
```bash
python make_nonlinear_benchmarks.py
```

## Run all
```bash
python eval_benchmarks.py
```

## Sensitivity analysis with P
```bash
python sweep_num_pairs.py
```

## Make Figure 1
```bash
python intro_fig_schematic.py
```
