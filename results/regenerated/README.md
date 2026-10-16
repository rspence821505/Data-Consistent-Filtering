# Regenerated results (Level 2 output)

This directory is filled by rerunning the experiments and is ignored by git.

```
results/regenerated/
├── raw/lorenz96/          raw, resumable checkpoints written by scripts/run_all_experiments.py
│   ├── baseline_extended/     Figures 1-2
│   ├── sensitivity/           Figure 3 (ensemble-size sweep)
│   └── diagnostics/           Figures 2, 4, 5 (coverage, hierarchical, (N, kappa))
├── lorenz96/*.npz         processed results, same schema as results/published/lorenz96/
└── shallow_water/         inlet tables rebuilt from the published run outputs
                           (scripts/extract_results.py --shallow-water)
```

Draw the figures from these results (Figures 6-8, which have no Level 2, are
always drawn from the published run outputs) with

```bash
python scripts/reproduce_figures.py --results regenerated
```

and compare them with the published numbers with

```bash
python scripts/verify_results.py
```

Nothing ever writes to `results/published/`.
