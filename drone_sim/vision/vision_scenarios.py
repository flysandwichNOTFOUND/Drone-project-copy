"""Synthetic image scenarios for the wire-connection detector.

Each scenario describes one synthetic test image that drone_sim/vision/run_wire_detector.py
renders and evaluates.  The approach mirrors tools/pid_scenarios.py: a frozen
dataclass defines every parameter; TRAINING_SCENARIOS are used during threshold
search; VALIDATION_SCENARIOS are held out until final evaluation.

Synthetic image generator
-------------------------
render_scenario(scenario) → uint8 RGB array (H×W×3)

The generator draws wires and plug bodies directly onto a NumPy array so that
no image library is required.  This keeps the dependency footprint identical to
the rest of the project (NumPy only).

Detection ground truth
----------------------
Each scenario declares expected_plugs and expected_wires so that run_workflow
can compute precision / recall metrics.
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen = True)
class WireScenario:
    name: str
    seed: int
    image_size: Tuple[int, int] = (128, 128)        # (height, width) in pixels
    background: str = "light"                        # "light" | "dark" | "grey"
    # Wire configuration
    wire_colours: Tuple[str, ...] = ("red", "black")
    wire_count: int = 2
    wire_thickness: int = 2
    # Plug / connector configuration
    plug_type: str = "xt_connector"                  # "xt_connector" | "crimp_plug" | "solder_joint"
    plug_count: int = 1
    # Noise and blur levels: 0.0 = clean, 1.0 = heavy
    noise_level: float = 0.05
    blur_radius: int = 0                             # 0 = no blur; 1+ = box-blur passes
    # Ground truth (used for evaluation only)
    expected_plugs: int = 1
    expected_wires: int = 2


# ---------------------------------------------------------------------------
# Training scenarios – used during threshold / parameter search
# ---------------------------------------------------------------------------

TRAINING_SCENARIOS = (
    WireScenario(
        name           = "train_xt60_clean",
        seed           = 10,
        image_size     = (128, 128),
        background     = "light",
        wire_colours   = ("red", "black"),
        wire_count     = 2,
        plug_type      = "xt_connector",
        plug_count     = 1,
        noise_level    = 0.03,
        blur_radius    = 0,
        expected_plugs = 1,
        expected_wires = 2,
    ),
    WireScenario(
        name           = "train_crimp_noisy",
        seed           = 11,
        image_size     = (128, 128),
        background     = "grey",
        wire_colours   = ("red", "black"),
        wire_count     = 2,
        plug_type      = "crimp_plug",
        plug_count     = 2,
        noise_level    = 0.12,
        blur_radius    = 0,
        expected_plugs = 2,
        expected_wires = 2,
    ),
    WireScenario(
        name           = "train_multi_wire",
        seed           = 12,
        image_size     = (160, 160),
        background     = "light",
        wire_colours   = ("red", "black", "yellow", "blue"),
        wire_count     = 4,
        plug_type      = "crimp_plug",
        plug_count     = 4,
        noise_level    = 0.05,
        blur_radius    = 0,
        expected_plugs = 4,
        expected_wires = 4,
    ),
)

# ---------------------------------------------------------------------------
# Held-out validation scenarios – never used to guide threshold search
# ---------------------------------------------------------------------------

VALIDATION_SCENARIOS = (
    WireScenario(
        name           = "val_dark_background",
        seed           = 101,
        image_size     = (128, 128),
        background     = "dark",
        wire_colours   = ("red", "yellow"),
        wire_count     = 2,
        plug_type      = "xt_connector",
        plug_count     = 1,
        noise_level    = 0.08,
        blur_radius    = 1,
        expected_plugs = 1,
        expected_wires = 2,
    ),
    WireScenario(
        name           = "val_solder_joints",
        seed           = 102,
        image_size     = (128, 128),
        background     = "grey",
        wire_colours   = ("red", "black"),
        wire_count     = 3,
        plug_type      = "solder_joint",
        plug_count     = 3,
        noise_level    = 0.10,
        blur_radius    = 0,
        expected_plugs = 3,
        expected_wires = 3,
    ),
    WireScenario(
        name           = "val_battery_cable_pair",
        seed           = 103,
        image_size     = (160, 160),
        background     = "light",
        wire_colours   = ("red", "black"),
        wire_count     = 2,
        plug_type      = "xt_connector",
        plug_count     = 2,
        noise_level    = 0.06,
        blur_radius    = 1,
        expected_plugs = 2,
        expected_wires = 2,
    ),
)

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

_WIRE_COLOURS = {
    "red":    (220,  40,  40),
    "black":  ( 20,  20,  20),
    "yellow": (220, 210,  20),
    "blue":   ( 30,  60, 220),
    "green":  ( 30, 180,  50),
    "white":  (230, 230, 230),
}

_BACKGROUND_COLOURS = {
    "light": (240, 238, 230),
    "grey":  (140, 140, 140),
    "dark":  ( 35,  35,  35),
}

# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _draw_line(canvas, r0, c0, r1, c1, colour, thickness = 2):
    """Draw a thick anti-aliased line using Bresenham + dilation."""
    h, w = canvas.shape[:2]
    dr   = abs(r1 - r0)
    dc   = abs(c1 - c0)
    sr   = 1 if r0 < r1 else -1
    sc   = 1 if c0 < c1 else -1
    err  = dr - dc
    r, c = r0, c0
    t    = thickness // 2

    while True:
        for dr2 in range(-t, t + 1):
            for dc2 in range(-t, t + 1):
                nr, nc = r + dr2, c + dc2
                if 0 <= nr < h and 0 <= nc < w:
                    canvas[nr, nc] = colour
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r   += sr
        if e2 < dr:
            err += dr
            c   += sc


def _draw_rect(canvas, r0, c0, r1, c1, colour):
    """Fill a rectangle with the given colour."""
    h, w = canvas.shape[:2]
    r0 = int(np.clip(r0, 0, h - 1))
    r1 = int(np.clip(r1, 0, h - 1))
    c0 = int(np.clip(c0, 0, w - 1))
    c1 = int(np.clip(c1, 0, w - 1))
    canvas[r0:r1 + 1, c0:c1 + 1] = colour


def _draw_ellipse(canvas, cr, cc, hr, hc, colour):
    """Rasterise a filled axis-aligned ellipse."""
    h, w = canvas.shape[:2]
    for dr in range(-hr, hr + 1):
        for dc in range(-hc, hc + 1):
            if (dr / max(hr, 1)) ** 2 + (dc / max(hc, 1)) ** 2 <= 1.0:
                r, c = int(cr) + dr, int(cc) + dc
                if 0 <= r < h and 0 <= c < w:
                    canvas[r, c] = colour


def _box_blur(canvas, passes = 1):
    """Simple box-blur applied `passes` times to simulate camera blur."""
    k = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype = float) / 9.0
    result = canvas.astype(float)
    for _ in range(passes):
        padded = np.pad(result, ((1, 1), (1, 1), (0, 0)), mode = "edge")
        blurred = np.zeros_like(result)
        for dr in range(3):
            for dc in range(3):
                blurred += k[dr, dc] * padded[dr:dr + result.shape[0],
                                               dc:dc + result.shape[1], :]
        result = blurred
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Plug drawing helpers
# ---------------------------------------------------------------------------

PLUG_COLOUR = (210, 210, 210)    # shiny metal / silver
SOLDER_COLOUR = (180, 160,  60)  # tin-solder golden


def _draw_plug(canvas, row, col, plug_type, rng):
    if plug_type == "xt_connector":
        size = rng.integers(10, 16)
        _draw_rect(canvas, row - size // 2, col - size // 2,
                   row + size // 2, col + size // 2, PLUG_COLOUR)
    elif plug_type == "crimp_plug":
        hr = rng.integers(4, 7)
        hc = rng.integers(8, 14)
        _draw_ellipse(canvas, row, col, hr, hc, PLUG_COLOUR)
    else:  # solder_joint
        hr = rng.integers(3, 5)
        _draw_ellipse(canvas, row, col, hr, hr, SOLDER_COLOUR)


# ---------------------------------------------------------------------------
# Public: render a scenario to a uint8 RGB image
# ---------------------------------------------------------------------------

def render_scenario(scenario: WireScenario) -> np.ndarray:
    """Render a synthetic wire/plug image for the given scenario.

    Returns
    -------
    np.ndarray, dtype uint8, shape (H, W, 3)
    """
    rng    = np.random.default_rng(scenario.seed)
    h, w   = scenario.image_size
    bg     = _BACKGROUND_COLOURS.get(scenario.background, (240, 238, 230))
    canvas = np.full((h, w, 3), bg, dtype = np.uint8)

    colour_names = list(scenario.wire_colours)
    n_wires      = min(scenario.wire_count, len(colour_names))
    wire_ends    = []     # collect endpoints to attach plugs

    for i in range(n_wires):
        colour_name = colour_names[i % len(colour_names)]
        colour      = _WIRE_COLOURS.get(colour_name, (80, 80, 80))
        margin      = 10
        r0 = int(rng.integers(margin, h - margin))
        c0 = int(rng.integers(margin, w // 2))
        r1 = int(rng.integers(margin, h - margin))
        c1 = int(rng.integers(w // 2, w - margin))
        _draw_line(canvas, r0, c0, r1, c1, colour, scenario.wire_thickness)
        wire_ends.append((r1, c1))   # right-hand end gets plug

    # Draw plugs at (some of) the wire ends.
    n_plugs = min(scenario.plug_count, len(wire_ends))
    for idx in range(n_plugs):
        pr, pc = wire_ends[idx]
        _draw_plug(canvas, pr, pc, scenario.plug_type, rng)

    # Add Gaussian noise.
    if scenario.noise_level > 0:
        noise  = rng.normal(0, scenario.noise_level * 255, canvas.shape)
        canvas = np.clip(canvas.astype(float) + noise, 0, 255).astype(np.uint8)

    # Optional box blur.
    if scenario.blur_radius > 0:
        canvas = _box_blur(canvas, passes = scenario.blur_radius)

    return canvas


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def detection_metrics(result: dict, scenario: WireScenario) -> dict:
    """Compute plug precision/recall and wire-count error for one result.

    Parameters
    ----------
    result   : dict returned by WireConnectionDetector.detect()
    scenario : the WireScenario that produced the image

    Returns
    -------
    dict with plug_precision, plug_recall, wire_count_error, score
    """
    found_plugs  = result["plug_count"]
    found_wires  = result["wire_count"]
    expected_p   = max(scenario.expected_plugs, 1)
    expected_w   = max(scenario.expected_wires, 1)

    precision  = min(found_plugs, expected_p) / max(found_plugs, 1)
    recall     = min(found_plugs, expected_p) / expected_p
    wire_error = abs(found_wires - scenario.expected_wires) / expected_w
    # Combined score: lower is better (0 = perfect)
    score      = (1 - precision) + (1 - recall) + wire_error
    return {
        "plug_precision":   round(precision, 4),
        "plug_recall":      round(recall, 4),
        "wire_count_error": round(wire_error, 4),
        "score":            round(score, 4),
    }


def evaluate_suite(detector, scenarios):
    """Run the detector on every scenario; return aggregate results.

    Parameters
    ----------
    detector  : WireConnectionDetector instance
    scenarios : iterable of WireScenario

    Returns
    -------
    dict with score (mean, lower=better), success_rate, cases list
    """
    cases = []
    for scenario in scenarios:
        image   = render_scenario(scenario)
        result  = detector.detect(image)
        metrics = detection_metrics(result, scenario)
        success = (metrics["plug_precision"] >= 0.5
                   and metrics["plug_recall"]     >= 0.5)
        cases.append({
            "scenario":     scenario.name,
            "overall_score":result["overall_score"],
            "plug_count":   result["plug_count"],
            "wire_count":   result["wire_count"],
            "warnings":     result["warnings"],
            **metrics,
            "success":      success,
        })

    scores = [c["score"] for c in cases]
    return {
        "score":        float(np.mean(scores) + 0.25 * np.max(scores)),
        "success_rate": float(np.mean([c["success"] for c in cases])),
        "cases":        cases,
    }
