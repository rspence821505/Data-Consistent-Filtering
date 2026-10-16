"""
Pixel-level comparison of a regenerated figure with the PNG used in the paper.

PNG files are compared as decoded pixels, not as bytes: the byte stream also
carries metadata (for example the Matplotlib version) that has nothing to do
with what is drawn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.int16)


def compare_png(generated: Path, reference: Path) -> dict:
    """Return size agreement and pixel statistics (channel values 0-255)."""
    a, b = _pixels(generated), _pixels(reference)
    result = {
        "generated_size": tuple(a.shape[1::-1]),
        "reference_size": tuple(b.shape[1::-1]),
        "same_size": a.shape == b.shape,
    }
    if not result["same_size"]:
        result.update(identical=False, max_abs_diff=None, rms_diff=None,
                      fraction_pixels_differing=None)
        return result
    diff = np.abs(a - b)
    differing = diff.max(axis=2) > 0
    result.update(
        identical=bool(not differing.any()),
        max_abs_diff=int(diff.max()),
        rms_diff=float(np.sqrt(np.mean(diff.astype(float) ** 2))),
        fraction_pixels_differing=float(differing.mean()),
    )
    return result


def describe(result: dict) -> str:
    if not result["same_size"]:
        return (f"DIFFERENT SIZE: {result['generated_size']} vs reference "
                f"{result['reference_size']}")
    if result["identical"]:
        return f"pixel-identical to the paper figure ({result['generated_size'][0]}x{result['generated_size'][1]})"
    return (f"differs from the paper figure: {100 * result['fraction_pixels_differing']:.3f}% of "
            f"pixels, max channel difference {result['max_abs_diff']}/255, "
            f"RMS {result['rms_diff']:.3f}")
