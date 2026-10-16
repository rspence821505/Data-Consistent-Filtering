"""
Reduction of raw experiment outputs to the processed numerical results that the
figure scripts read.

This package is the interface between the two reproducibility levels described
in the README:

* Level 2 (the experiment drivers in ``src/experiments`` and ``shallow_water/``)
  writes raw, resumable checkpoints.
* ``src.processing`` reduces those checkpoints to a small, fixed set of files
  (``results/<set>/lorenz96/*.npz`` and ``results/<set>/shallow_water/*``).
* Level 1 (``src.figures``) reads only those processed files.

The published files in ``results/published/`` were produced by exactly this
code from the original raw checkpoints, so regenerated results pass through the
same reduction as the published ones.
"""
