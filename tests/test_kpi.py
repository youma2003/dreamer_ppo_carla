"""KPI / variant-comparison validation — runs with NO CARLA installed.

Covers evaluation.kpi: aggregation, the VRU-first scores, log round-trip via
the real Logger, ranking, and the text table.

Run with:  python tests/test_kpi.py
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from training.logger import Logger
from evaluation.kpi import (
    KPIWeights, compute_kpis, load_log, compare_variants,
    format_comparison_table,
)


def ok(name, result=""):
    print(f"✅ {name} : {result}")


def _row(ret, route, vru_coll=0, vru_nm=0, min_ttc=3.0, veh_coll=0, lane_dep=0):
    return {
        "return": ret, "route_completion": route,
        "vru_collisions": vru_coll, "vru_near_misses": vru_nm,
        "min_ttc_vru": min_ttc, "avg_distance_to_vru": 8.0,
        "vehicle_collisions": veh_coll, "vehicle_near_misses": 0,
        "rear_incidents": 0, "lane_departures": lane_dep,
        "lane_change_success_rate": 1.0, "lane_changes_unsafe_prevented": 0,
    }


def test_compute_basic():
    rows = [_row(10.0, 0.5), _row(20.0, 1.0)]
    kpi = compute_kpis(rows, tail=None)
    assert kpi["episodes_evaluated"] == 2
    assert abs(kpi["mean_return"] - 15.0) < 1e-9
    assert abs(kpi["mean_route_completion"] - 0.75) < 1e-9
    assert abs(kpi["success_rate"] - 0.5) < 1e-9  # only the route=1.0 episode
    ok("compute_basic", f"return={kpi['mean_return']}, success={kpi['success_rate']}")


def test_perfect_safety_scores_100():
    rows = [_row(50.0, 1.0, min_ttc=5.0) for _ in range(4)]
    kpi = compute_kpis(rows, tail=None)
    assert kpi["vru_safety_score"] == 100.0
    assert kpi["vehicle_safety_score"] == 100.0
    assert kpi["comfort_score"] == 100.0          # 0 lane departures
    assert kpi["composite_score"] == 100.0
    ok("perfect_safety", "all scores 100")


def test_vru_collision_dominates():
    """A high-return variant that hits VRUs must score below a safe one."""
    unsafe = [_row(100.0, 1.0, vru_coll=1) for _ in range(4)]
    safe = [_row(40.0, 0.7) for _ in range(4)]
    res = compare_variants({"unsafe": unsafe, "safe": safe}, tail=None)
    ranked = list(res.keys())
    assert ranked[0] == "safe", ranked
    assert res["unsafe"]["vru_safety_score"] < res["safe"]["vru_safety_score"]
    # 1 collision/ep * 45 -> 55
    assert abs(res["unsafe"]["vru_safety_score"] - 55.0) < 1e-6
    ok("vru_collision_dominates", f"safe ranked first; unsafe vru="
       f"{res['unsafe']['vru_safety_score']:.0f}")


def test_low_ttc_penalised():
    safe = compute_kpis([_row(10.0, 1.0, min_ttc=3.0)], tail=None)
    risky = compute_kpis([_row(10.0, 1.0, min_ttc=0.5)], tail=None)
    assert risky["vru_safety_score"] < safe["vru_safety_score"]
    # (2.0 - 0.5) * 10 = 15 points lost
    assert abs(risky["vru_safety_score"] - 85.0) < 1e-6
    ok("low_ttc_penalised", f"risky={risky['vru_safety_score']:.0f}")


def test_tail_window():
    rows = [_row(0.0, 0.0, vru_coll=1) for _ in range(8)]  # bad early
    rows += [_row(50.0, 1.0) for _ in range(2)]            # good recent
    kpi = compute_kpis(rows, tail=0.2)                     # last 20% = 2 rows
    assert kpi["episodes_evaluated"] == 2
    assert kpi["vru_collisions_per_ep"] == 0.0
    ok("tail_window", "evaluated only converged tail")


def test_log_roundtrip_and_table():
    d = tempfile.mkdtemp()
    logger = Logger(log_dir=d, filename="v.csv")
    for ep in range(3):
        logger.log(ep, _row(10.0 * ep, 1.0, vru_nm=ep))
    logger.close()
    rows = load_log(os.path.join(d, "v.csv"))
    assert len(rows) == 3
    kpi = compute_kpis(rows, tail=None)
    assert abs(kpi["vru_near_misses_per_ep"] - 1.0) < 1e-9  # (0+1+2)/3
    res = compare_variants({"v": os.path.join(d, "v.csv")}, tail=None)
    table = format_comparison_table(res)
    assert "COMPOSITE SCORE" in table
    assert "VRU SAFETY" in table
    ok("log_roundtrip_and_table", "CSV -> KPIs -> table")


def test_weights_tunable():
    rows = [_row(10.0, 1.0, vru_coll=1)]
    strict = compute_kpis(rows, weights=KPIWeights(w_vru_collision=90.0), tail=None)
    lenient = compute_kpis(rows, weights=KPIWeights(w_vru_collision=10.0), tail=None)
    assert strict["vru_safety_score"] < lenient["vru_safety_score"]
    ok("weights_tunable", f"strict={strict['vru_safety_score']:.0f} "
       f"lenient={lenient['vru_safety_score']:.0f}")


def main():
    print("Running KPI / comparison tests (no CARLA needed)...\n")
    test_compute_basic()
    test_perfect_safety_scores_100()
    test_vru_collision_dominates()
    test_low_ttc_penalised()
    test_tail_window()
    test_log_roundtrip_and_table()
    test_weights_tunable()
    print("\n✅ ALL KPI TESTS PASSED")


if __name__ == "__main__":
    main()
