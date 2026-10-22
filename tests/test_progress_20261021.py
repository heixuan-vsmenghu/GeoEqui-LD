"""Checks for the public aggregate report; no models or medical data are loaded."""

import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "progress_20261021"
METRICS = ("MRE_PSR", "MRE_PSL", "MRE_FHT", "MRE_ALL", "Delta_AoP")


def _data():
    return json.loads((REPORT / "metrics.json").read_text(encoding="utf-8"))


def _close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9)


def _metrics(value):
    assert set(METRICS) <= set(value)
    assert all(math.isfinite(value[key]) for key in METRICS)
    _close(value["MRE_ALL"], sum(value[key] for key in METRICS[:3]) / 3)


def test_public_report_exists():
    assert (REPORT / "REPORT.pdf").read_bytes().startswith(b"%PDF-")
    for name in ("REPORT.md", "RESULTS.md", "PROTOCOLS.md", "LIMITATIONS.md"):
        assert len((REPORT / name).read_text(encoding="utf-8")) > 100


def test_scope_and_primary_identity():
    data = _data()
    assert data["results_through"] == "2026-10-20"
    assert data["evaluation"]["new_Testing_runs"] == 0
    assert data["evaluation"]["new_bootstrap_runs"] == 0
    assert data["evaluation"]["validation_is_repeated_development_set"] is True
    assert data["evaluation"]["spacing_mm_available"] is False
    primary = data["external_primary"]
    assert len(primary) == 4
    for method, stage in (("STLD", "R4_MAIN"), ("PIPNet-GSSL", "G5_STD")):
        rows = [row for row in primary if row["method"] == method]
        assert {row["seed"] for row in rows} == {2026101801, 2026101901}
        assert all(row["stage"] == stage for row in rows)
        for split, expected_n in (("Validation", 100), ("Training", 300)):
            for row in rows:
                assert row[split]["sample_count"] == expected_n
                assert (
                    row[split]["AoP_valid_count"]
                    + row[split]["AoP_invalid_count"] == expected_n
                )
                _metrics(row[split]["metrics"])
            for key in METRICS:
                values = [row[split]["metrics"][key] for row in rows]
                summary = data["external_two_seed_summary"][method][split][key]
                _close(summary["mean"], statistics.mean(values))
                _close(summary["sample_sd"], statistics.stdev(values))
                assert summary["n_seeds"] == 2


def test_all_external_stages_are_aggregates():
    rows = _data()["external_stages"]
    assert len(rows) == 32
    assert len({(r["method"], r["seed"], r["stage"]) for r in rows}) == 32
    for row in rows:
        for split in ("Validation", "Training"):
            _metrics(row[split]["metrics"])
            for tail in row[split]["tail"].values():
                assert tail["n"] == row[split]["sample_count"]
                assert tail["median"] <= tail["p95"] <= tail["max"]


def test_initialization_pairs_and_summary():
    data = _data()
    pairs = data["initialization_pairs"]
    assert {row["seed"] for row in pairs} == {2026101801, 2026101901}
    for split in ("Validation", "Training"):
        for key in METRICS:
            differences = []
            for row in pairs:
                assert row["updates_each"] == 1500
                delta = (
                    row["ImageNet"][split]["metrics"][key]
                    - row["Random_backbone"][split]["metrics"][key]
                )
                _close(delta, row["ImageNet_minus_Random"][split][key])
                differences.append(delta)
            summary = data["initialization_summary"][split][key]
            _close(summary["mean"], statistics.mean(differences))
            _close(summary["sample_sd"], statistics.stdev(differences))


def test_course_deltas_use_fixed_start_and_end():
    data = _data()
    index = {
        (row["method"], row["seed"], row["stage"]): row
        for row in data["external_stages"]
    }
    assert len(data["course_differences"]) == 4
    for row in data["course_differences"]:
        start = index[(row["method"], row["seed"], row["start_stage"])]
        end = index[(row["method"], row["seed"], row["end_stage"])]
        for split in ("Validation", "Training"):
            for key in METRICS:
                _close(
                    end[split]["metrics"][key] - start[split]["metrics"][key],
                    row["end_minus_start"][split][key],
                )


def test_geo_factorial_and_structure():
    data = _data()
    assert len(data["geo_factorial"]) == 8
    for condition in ("S", "SG", "SP", "SGP"):
        rows = [r for r in data["geo_factorial"] if r["condition"] == condition]
        assert {r["seed"] for r in rows} == {43, 44}
        for row in rows:
            _metrics(row["metrics"])
        for key in METRICS:
            _close(
                statistics.mean(r["metrics"][key] for r in rows),
                data["geo_factorial_means"][condition][key],
            )
    assert len(data["geo_structure"]) == 20
    for row in data["geo_structure"]:
        _metrics(row["metrics"])
    assert len(data["fh_freshpair"]) == 8
    for row in data["fh_freshpair"]:
        _metrics(row["metrics"])


def test_cdf_keeps_seeds_and_denominators_separate():
    data = json.loads((REPORT / "cdf_counts.json").read_text(encoding="utf-8"))
    assert len(data["curves"]) == 4
    for curve in data["curves"]:
        thresholds = curve["thresholds_original_px"]
        assert thresholds == list(range(0, 257, 2))
        for key in ("PSR", "PSL", "FHT", "ALL_image_mean"):
            entry = curve[key]
            counts = entry["counts_le"]
            assert entry["n"] == 100
            assert len(counts) == len(thresholds)
            assert counts == sorted(counts)
            assert all(type(value) is int and 0 <= value <= 100 for value in counts)
            assert counts[-1] + entry["above_max"] == 100


def test_mainline_revision_uses_structure_before_external():
    text = (REPORT / "REPORT.md").read_text(encoding="utf-8")
    assert text.index("1　结构消融") < text.index("5　外部方法参照")
    assert "共同未训练起点" in text
    assert "目前尚未执行" in text
    assert "本阶段可汇报" not in text
    for variant in ("去PS增强", "去FH增强", "无增强・独立头", "无增强・共享头"):
        assert variant in text
    data = _data()
    for variant in ("FULL", "NO_PS", "NO_FH", "PLAIN_SPLIT", "PLAIN_SHARED"):
        rows = [row for row in data["geo_structure"]
                if row["variant"] == variant and row["phase"] == "semi"]
        assert len(rows) == 2
        for key in METRICS:
            value = statistics.mean(row["metrics"][key] for row in rows)
            assert f"{value:.3f}" in text


def test_p1_midpoints_have_distinct_public_identity():
    text = (REPORT / "RESULTS.md").read_text(encoding="utf-8")
    assert "新更新步数" in text and "总监督遍数" in text
    assert "长监督中间观察" in text and "长监督固定终点" in text
    data = _data()
    long_rows = [row for row in data["geo_supervised_diagnostic"]
                 if row["condition"] == "supervised_total_300_epochs"]
    assert len(long_rows) == 6
    assert {row["new_updates"] for row in long_rows} == {21000, 28500, 36000}
    assert sum(row["new_updates"] == 36000 for row in long_rows) == 2


def test_request_alignment_does_not_call_first_round_a_solution():
    text = (REPORT / "REQUEST_ALIGNMENT.md").read_text(encoding="utf-8")
    assert "基础上限与根因" in text
    assert "新初始化" not in text or "未执行" in text
    assert "近期医学SOTA覆盖" in text
    assert "未执行" in text
