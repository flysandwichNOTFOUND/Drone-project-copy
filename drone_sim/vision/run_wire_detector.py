"""Wire-connection detector workflow.

Mirrors the structure of tools/retune_pid.py:

    run_workflow()   – search detector thresholds on training scenarios,
                       then validate on held-out scenarios.
    main()           – CLI entry-point.

Outputs (written to outputs/)
------------------------------
wire_detector_report.json   Full structured report (schema_version 1).
wire_detector_config.json   Accepted thresholds only (lightweight config).

Usage
-----
    python -m drone_sim.vision.run_wire_detector
    python -m drone_sim.vision.run_wire_detector --trials 30 --save-images
"""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from drone_sim.configuration.project_paths import output_path
from drone_sim.vision.detector import WireConnectionDetector
from drone_sim.vision.vision_scenarios import (
    TRAINING_SCENARIOS,
    VALIDATION_SCENARIOS,
    evaluate_suite,
    render_scenario,
)


# ---------------------------------------------------------------------------
# Search space: [min, max] for each tuneable threshold
# ---------------------------------------------------------------------------

# [min_plug_area, min_plug_brightness, wire_neighbourhood_radius]
SEARCH_BOUNDS = np.array(
    [[4,   30],    # min_plug_area        (px²)
     [0.3,  0.75], # min_plug_brightness  (0–1)
     [6,   20]],   # wire_neighbourhood_radius (px)
    dtype = float,
)

TRIALS_PER_SEARCH  = 40    # random threshold candidates evaluated per seed
DEFAULT_SEEDS      = (7, 29)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detector(params):
    """Instantiate a WireConnectionDetector from a parameter array/dict."""
    if isinstance(params, dict):
        return WireConnectionDetector(
            min_plug_area              = int(round(params["min_plug_area"])),
            min_plug_brightness        = float(params["min_plug_brightness"]),
            wire_neighbourhood_radius  = int(round(params["wire_neighbourhood_radius"])),
        )
    return WireConnectionDetector(
        min_plug_area              = int(round(float(params[0]))),
        min_plug_brightness        = float(params[1]),
        wire_neighbourhood_radius  = int(round(float(params[2]))),
    )


def _params_to_dict(params):
    return {
        "min_plug_area":             int(round(float(params[0]))),
        "min_plug_brightness":       round(float(params[1]), 6),
        "wire_neighbourhood_radius": int(round(float(params[2]))),
    }


def _default_params():
    return np.array([8.0, 0.50, 12.0], dtype = float)


def _random_params(rng, bounds = SEARCH_BOUNDS):
    return rng.uniform(bounds[:, 0], bounds[:, 1])


def _json_safe(obj):
    """Recursively convert NumPy scalars / arrays to plain Python for JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# ---------------------------------------------------------------------------
# Single-seed random search
# ---------------------------------------------------------------------------

def _search_thresholds(baseline_result, seed, trials = TRIALS_PER_SEARCH):
    """Random search over SEARCH_BOUNDS; returns best params and training result."""
    rng       = np.random.default_rng(seed)
    best_p    = _default_params()
    best_res  = baseline_result

    for _ in range(trials):
        candidate_p = _random_params(rng)
        detector    = _make_detector(candidate_p)
        result      = evaluate_suite(detector, TRAINING_SCENARIOS)
        if (result["score"] < best_res["score"]
                and result["success_rate"] >= baseline_result["success_rate"]):
            best_p, best_res = candidate_p, result

    return best_p, best_res


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run_workflow(seeds = DEFAULT_SEEDS, save_images = False, trials = TRIALS_PER_SEARCH):
    """Search detector thresholds, then gate on held-out validation.

    Parameters
    ----------
    seeds       : iterable of ints – random seeds for the threshold search.
    save_images : bool – if True, render and save scenario PNGs to outputs/.

    Returns
    -------
    dict  Full report (also written to outputs/wire_detector_report.json).
    """
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("Trials must be a positive integer.")
    seeds = list(seeds)
    if not seeds:
        raise ValueError("At least one search seed is required.")

    baseline_p      = _default_params()
    baseline_det    = _make_detector(baseline_p)
    baseline_train  = evaluate_suite(baseline_det, TRAINING_SCENARIOS)

    best_p, best_train = baseline_p.copy(), baseline_train
    searches = []

    for seed in seeds:
        candidate_p, candidate_train = _search_thresholds(baseline_train, seed, trials)
        searches.append({
            "seed":     seed,
            "params":   _params_to_dict(candidate_p),
            "training": candidate_train,
        })
        if (candidate_train["score"] < best_train["score"]
                and candidate_train["success_rate"] >= baseline_train["success_rate"]):
            best_p, best_train = candidate_p, candidate_train

    # Held-out validation – run only once, never used to guide search.
    best_det    = _make_detector(best_p)
    baseline_v  = evaluate_suite(baseline_det, VALIDATION_SCENARIOS)
    changed     = not np.allclose(best_p, baseline_p)
    candidate_v = evaluate_suite(best_det, VALIDATION_SCENARIOS) if changed else baseline_v

    accepted = bool(
        changed
        and candidate_v["success_rate"] == 1.0
        and candidate_v["score"] <= baseline_v["score"]
    )
    selected_p = best_p if accepted else baseline_p

    # Optional: save rendered scenario images.
    if save_images:
        _save_scenario_images()

    report = {
        "schema_version":    1,
        "random_seeds":      seeds,
        "trials_per_search": trials,
        "search_bounds":     SEARCH_BOUNDS,
        "baseline_params":   _params_to_dict(baseline_p),
        "candidate_params":  _params_to_dict(best_p),
        "selected_params":   _params_to_dict(selected_p),
        "baseline_training": baseline_train,
        "candidate_training":best_train,
        "baseline_validation":   baseline_v,
        "candidate_validation":  candidate_v,
        "candidate_changed":     changed,
        "candidate_accepted":    accepted,
        "searches":              searches,
        "training_scenarios":    [s.name for s in TRAINING_SCENARIOS],
        "validation_scenarios":  [s.name for s in VALIDATION_SCENARIOS],
        "scope": (
            "Wire/plug threshold search on synthetic images.  "
            "Held-out validation is a gate, not a search objective."
        ),
    }

    report_path = output_path("wire_detector_report.json")
    report_path.write_text(
        json.dumps(_json_safe(report), indent = 2, allow_nan = False),
        encoding = "utf-8",
    )

    config_path = output_path("wire_detector_config.json")
    config_path.write_text(
        json.dumps(_json_safe({
            "selected_params":   _params_to_dict(selected_p),
            "candidate_accepted": accepted,
            "source_report":     report_path.name,
        }), indent = 2, allow_nan = False),
        encoding = "utf-8",
    )

    print(f"Candidate accepted: {accepted}")
    print(f"Baseline  – training score={baseline_train['score']:.3f}, "
          f"success={baseline_train['success_rate']:.0%}")
    print(f"Candidate – training score={best_train['score']:.3f}, "
          f"success={best_train['success_rate']:.0%}")
    print(f"Baseline  – validation score={baseline_v['score']:.3f}, "
          f"success={baseline_v['success_rate']:.0%}")
    label = "Candidate" if changed else "Retained baseline"
    print(f"{label} – validation score={candidate_v['score']:.3f}, "
          f"success={candidate_v['success_rate']:.0%}")
    print(f"Saved {report_path}")
    return report


# ---------------------------------------------------------------------------
# Image saving helper
# ---------------------------------------------------------------------------

def _save_scenario_images():
    """Render all training and validation scenarios and write PNG files."""
    try:
        from PIL import Image as _PIL_Image
        _have_pil = True
    except ImportError:
        _have_pil = False

    for scenario in list(TRAINING_SCENARIOS) + list(VALIDATION_SCENARIOS):
        img_array = render_scenario(scenario)
        path = output_path(f"wire_scenario_{scenario.name}.png")
        if _have_pil:
            _PIL_Image.fromarray(img_array).save(str(path))
        else:
            # Fall back to PPM (no extra libraries needed).
            ppm_path = path.with_suffix(".ppm")
            h, w = img_array.shape[:2]
            with open(ppm_path, "wb") as fh:
                fh.write(f"P6\n{w} {h}\n255\n".encode())
                fh.write(img_array.tobytes())
            path = ppm_path
        print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(seeds = DEFAULT_SEEDS, trials = TRIALS_PER_SEARCH, save_images = False):
    return run_workflow(seeds = seeds, save_images = save_images, trials = trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Search and validate wire-connection detector thresholds.")
    parser.add_argument(
        "--seeds", type = int, nargs = "+", default = list(DEFAULT_SEEDS),
        help = "Random seeds for threshold search (default: 7 29).")
    parser.add_argument(
        "--trials", type = int, default = TRIALS_PER_SEARCH,
        help = f"Threshold candidates per seed (default: {TRIALS_PER_SEARCH}).")
    parser.add_argument(
        "--save-images", action = "store_true",
        help = "Render and save scenario images to outputs/.")
    args = parser.parse_args()
    main(tuple(args.seeds), args.trials, args.save_images)
