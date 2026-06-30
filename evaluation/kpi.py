"""KPIs for comparing the Dreamer variants, with VRU safety as the priority.

Every variant (PPO baseline, greedy Dreamer-PPO, SDBS Dreamer-PPO) writes the
same per-episode CSV via ``training.logger.Logger``. This module turns those
logs into a small, comparable set of Key Performance Indicators and combines
them into two headline scores:

    * ``vru_safety_score``  — 0..100, *higher is safer for pedestrians/cyclists*
    * ``composite_score``   — 0..100, safety-weighted overall driving quality

The weighting deliberately puts Vulnerable Road User (VRU) safety first: a
variant that completes more of the route but hits a pedestrian must never rank
above one that drives a little less far but keeps people safe. Vehicle safety
and ride comfort are secondary.

No pandas dependency — plain ``csv`` like ``plot_results.py``, so this runs
anywhere the trainer runs.
"""
import csv
import math
import os
from collections import OrderedDict
from dataclasses import dataclass

# A route is "successfully completed" once progress crosses this fraction.
SUCCESS_ROUTE_THRESHOLD = 0.95
# Default time-to-collision horizon (matches Config.tau_ttc); a per-episode
# min-TTC below this is unsafe and is penalised.
DEFAULT_TAU_TTC = 2.0


@dataclass
class KPIWeights:
    """Tunable weights for the headline composite scores.

    Defaults encode the project's safety hierarchy: VRU >> vehicle >> comfort.
    All ``w_*`` penalties are in *score points subtracted per unit of the
    metric* (e.g. ``w_vru_collision=45`` removes 45 points for every VRU
    collision per episode), so the scores stay interpretable.
    """

    # --- VRU safety score (primary) ---
    w_vru_collision: float = 45.0     # per VRU collision / episode
    w_vru_near_miss: float = 6.0      # per VRU near-miss / episode
    w_vru_low_ttc: float = 10.0       # per second of min-TTC below tau_ttc

    # --- composite score blend (must sum to 1.0) ---
    blend_vru_safety: float = 0.55    # VRU safety is the largest single share
    blend_progress: float = 0.25      # route completion
    blend_vehicle_safety: float = 0.12
    blend_comfort: float = 0.08

    # --- vehicle safety sub-score penalties ---
    w_vehicle_collision: float = 25.0
    w_vehicle_near_miss: float = 3.0
    w_rear_incident: float = 4.0


# --------------------------------------------------------------------------- #
# Log loading
# --------------------------------------------------------------------------- #
def load_log(path):
    """Read a training-log CSV into a list of dict rows (values as floats).

    Missing/blank cells become 0.0 so partial logs (e.g. a baseline run that
    never populated vehicle columns) still aggregate cleanly.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, newline="", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))
    rows = []
    for r in raw:
        row = {}
        for key, value in r.items():
            if key is None:
                continue
            try:
                row[key] = float(value) if value not in (None, "") else 0.0
            except (TypeError, ValueError):
                row[key] = 0.0
        rows.append(row)
    return rows


def _tail(rows, tail):
    """Return the last ``tail`` rows (the converged regime).

    ``tail=None`` keeps all rows; a fraction in (0, 1] keeps that share of the
    tail; an int >= 1 keeps that many rows.
    """
    if not rows or tail is None:
        return rows
    if 0 < tail <= 1:
        n = max(1, int(round(len(rows) * tail)))
    else:
        n = int(tail)
    return rows[-n:] if n < len(rows) else rows


def _mean(rows, key):
    if not rows:
        return 0.0
    return sum(r.get(key, 0.0) for r in rows) / len(rows)


def _rate(rows, key):
    """Fraction of episodes whose ``key`` is > 0 (e.g. collision rate)."""
    if not rows:
        return 0.0
    return sum(1 for r in rows if r.get(key, 0.0) > 0) / len(rows)


def _mean_positive(rows, key):
    """Mean over episodes where ``key`` is > 0 (e.g. min-TTC, which is 0 when
    no VRU was ever observed and would otherwise drag the average down)."""
    vals = [r[key] for r in rows if r.get(key, 0.0) > 0]
    return sum(vals) / len(vals) if vals else 0.0


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #
def _vru_safety_score(kpi, weights, tau_ttc):
    """0..100; starts perfect and subtracts safety penalties."""
    score = 100.0
    score -= weights.w_vru_collision * kpi["vru_collisions_per_ep"]
    score -= weights.w_vru_near_miss * kpi["vru_near_misses_per_ep"]
    ttc = kpi["mean_min_ttc_vru"]
    if ttc > 0:  # 0 means "no VRU observed" -> no TTC penalty
        score -= weights.w_vru_low_ttc * max(0.0, tau_ttc - ttc)
    return max(0.0, min(100.0, score))


def _vehicle_safety_score(kpi, weights):
    score = 100.0
    score -= weights.w_vehicle_collision * kpi["vehicle_collisions_per_ep"]
    score -= weights.w_vehicle_near_miss * kpi["vehicle_near_misses_per_ep"]
    score -= weights.w_rear_incident * kpi["rear_incidents_per_ep"]
    return max(0.0, min(100.0, score))


def _comfort_score(kpi):
    """Map lane departures per episode to 0..100 (fewer is better).

    A smooth exponential decay: 0 departures -> 100, ~1/ep -> ~37.
    """
    return 100.0 * math.exp(-kpi["lane_departures_per_ep"])


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def compute_kpis(rows, weights=None, tail=0.25, tau_ttc=DEFAULT_TAU_TTC,
                 success_threshold=SUCCESS_ROUTE_THRESHOLD):
    """Aggregate a list of per-episode rows into a KPI dict.

    ``tail`` selects the converged tail of training (default: last 25% of
    episodes) so a variant is judged on its learned behaviour, not its early
    flailing. Pass ``tail=None`` to use every episode.

    Returns an ``OrderedDict`` grouped performance -> VRU -> vehicle ->
    quality -> headline scores, suitable for tabular display.
    """
    weights = weights or KPIWeights()
    window = _tail(rows, tail)
    n = len(window)

    kpi = OrderedDict()
    kpi["episodes_evaluated"] = n

    # --- performance ---
    kpi["mean_return"] = _mean(window, "return")
    kpi["mean_route_completion"] = _mean(window, "route_completion")
    kpi["success_rate"] = (
        sum(1 for r in window if r.get("route_completion", 0.0) >= success_threshold)
        / n if n else 0.0)

    # --- VRU safety (PRIMARY) ---
    kpi["vru_collisions_per_ep"] = _mean(window, "vru_collisions")
    kpi["vru_collision_rate"] = _rate(window, "vru_collisions")
    kpi["vru_near_misses_per_ep"] = _mean(window, "vru_near_misses")
    kpi["mean_min_ttc_vru"] = _mean_positive(window, "min_ttc_vru")
    kpi["mean_distance_to_vru"] = _mean_positive(window, "avg_distance_to_vru")

    # --- vehicle safety (secondary) ---
    kpi["vehicle_collisions_per_ep"] = _mean(window, "vehicle_collisions")
    kpi["vehicle_near_misses_per_ep"] = _mean(window, "vehicle_near_misses")
    kpi["rear_incidents_per_ep"] = _mean(window, "rear_incidents")

    # --- driving quality ---
    kpi["lane_departures_per_ep"] = _mean(window, "lane_departures")
    kpi["lane_change_success_rate"] = _mean(window, "lane_change_success_rate")
    kpi["unsafe_lane_changes_prevented_per_ep"] = _mean(
        window, "lane_changes_unsafe_prevented")

    # --- headline scores ---
    kpi["vru_safety_score"] = _vru_safety_score(kpi, weights, tau_ttc)
    kpi["vehicle_safety_score"] = _vehicle_safety_score(kpi, weights)
    kpi["comfort_score"] = _comfort_score(kpi)
    progress_score = 100.0 * max(0.0, min(1.0, kpi["mean_route_completion"]))
    kpi["composite_score"] = (
        weights.blend_vru_safety * kpi["vru_safety_score"]
        + weights.blend_progress * progress_score
        + weights.blend_vehicle_safety * kpi["vehicle_safety_score"]
        + weights.blend_comfort * kpi["comfort_score"])
    return kpi


def compute_kpis_from_log(path, **kwargs):
    """Convenience: ``load_log`` then ``compute_kpis``."""
    return compute_kpis(load_log(path), **kwargs)


def compare_variants(named_logs, weights=None, **kwargs):
    """Compute KPIs for several variants.

    ``named_logs`` maps a label -> CSV path (or a label -> list-of-rows, so
    in-memory ``history`` from a trainer can be compared without writing files).
    Returns an ``OrderedDict`` label -> kpi dict, ordered best-VRU-safety first.
    """
    weights = weights or KPIWeights()
    results = OrderedDict()
    for label, source in named_logs.items():
        rows = load_log(source) if isinstance(source, str) else list(source)
        results[label] = compute_kpis(rows, weights=weights, **kwargs)
    # Rank by VRU safety first, then composite — the project's priority order.
    ranked = OrderedDict(
        sorted(results.items(),
               key=lambda kv: (kv[1]["vru_safety_score"],
                               kv[1]["composite_score"]),
               reverse=True))
    return ranked


# --------------------------------------------------------------------------- #
# Display
# --------------------------------------------------------------------------- #
# (label, format) for the rows of the comparison table, in display order.
_TABLE_ROWS = [
    ("Episodes evaluated", "episodes_evaluated", "{:.0f}"),
    ("-- PERFORMANCE", None, None),
    ("Mean return", "mean_return", "{:.1f}"),
    ("Route completion", "mean_route_completion", "{:.1%}"),
    ("Success rate", "success_rate", "{:.1%}"),
    ("-- VRU SAFETY (primary)", None, None),
    ("VRU collisions / ep", "vru_collisions_per_ep", "{:.3f}"),
    ("VRU collision rate", "vru_collision_rate", "{:.1%}"),
    ("VRU near-misses / ep", "vru_near_misses_per_ep", "{:.2f}"),
    ("Mean min TTC-VRU (s)", "mean_min_ttc_vru", "{:.2f}"),
    ("Mean dist to VRU (m)", "mean_distance_to_vru", "{:.2f}"),
    ("-- VEHICLE SAFETY", None, None),
    ("Vehicle collisions / ep", "vehicle_collisions_per_ep", "{:.3f}"),
    ("Vehicle near-misses / ep", "vehicle_near_misses_per_ep", "{:.2f}"),
    ("Rear incidents / ep", "rear_incidents_per_ep", "{:.2f}"),
    ("-- DRIVING QUALITY", None, None),
    ("Lane departures / ep", "lane_departures_per_ep", "{:.3f}"),
    ("Lane-change success", "lane_change_success_rate", "{:.1%}"),
    ("-- SCORES (higher=better)", None, None),
    ("VRU safety score", "vru_safety_score", "{:.1f}"),
    ("Vehicle safety score", "vehicle_safety_score", "{:.1f}"),
    ("Comfort score", "comfort_score", "{:.1f}"),
    ("COMPOSITE SCORE", "composite_score", "{:.1f}"),
]


def format_comparison_table(results):
    """Render ``compare_variants`` output as a fixed-width text table."""
    labels = list(results.keys())
    label_w = max([18] + [len(s) for s, _, _ in _TABLE_ROWS])
    col_w = max(12, *(len(l) for l in labels)) if labels else 12

    def fmt_cell(text):
        return f"{text:>{col_w}}"

    lines = []
    header = f"{'KPI':<{label_w}} " + " ".join(fmt_cell(l) for l in labels)
    sep = "=" * len(header)
    lines.append(sep)
    lines.append("DREAMER VARIANT COMPARISON (ranked by VRU safety)")
    lines.append(sep)
    lines.append(header)
    lines.append("-" * len(header))
    for name, key, fmt in _TABLE_ROWS:
        if key is None:  # section divider
            lines.append(f"{name:<{label_w}}")
            continue
        cells = []
        for label in labels:
            val = results[label].get(key, 0.0)
            cells.append(fmt_cell(fmt.format(val)))
        lines.append(f"{name:<{label_w}} " + " ".join(cells))
    lines.append(sep)
    if labels:
        best = labels[0]
        lines.append(f"Safest variant for VRUs: {best} "
                     f"(VRU safety {results[best]['vru_safety_score']:.1f}, "
                     f"composite {results[best]['composite_score']:.1f})")
        lines.append(sep)
    return "\n".join(lines)
