"""Constructed arrays only: no model, image, annotation or experiment inputs."""

import numpy as np

from geoequi_ld.fh_radius_postprocess import EPS, MODEL_IDS, combine


def test_fixed_radius_uses_mean_distances_to_the_geo_anchor():
    points = {
        "GEO_A": np.array([[[0.0, 0.0], [0.0, 2.0], [6.0, 8.0]]]),
        "GEO_B": np.array([[[2.0, 0.0], [2.0, 2.0], [8.0, 8.0]]]),
        "STLD1801": np.array([[[9.0, 9.0], [9.0, 8.0], [1.0, 2.0]]]),
        "STLD1901": np.array([[[-9.0, -9.0], [-9.0, -8.0], [1.0, -6.0]]]),
    }
    outputs, metadata = combine(points)

    # Geo anchor (1, 0), Geo direction (0.6, 0.8), STLD radii 2 and 6.
    np.testing.assert_allclose(outputs["CANDIDATE"][0, 2], [3.4, 3.2], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(outputs["STLD_ENS"][0, 2], [1.0, -2.0])
    assert metadata["fallback_count"] == 0
    assert metadata["PSR_PSL_exact_GEO_ENS"] is True


def test_ps_points_stay_exactly_geo_and_inputs_are_unchanged():
    points = {
        model: np.arange(12, dtype=np.float64).reshape(2, 3, 2) + offset
        for model, offset in zip(MODEL_IDS, (0.0, 2.0, 10.0, -5.0), strict=True)
    }
    before = {model: value.copy() for model, value in points.items()}
    expected_geo = (points["GEO_A"] + points["GEO_B"]) / 2
    outputs, metadata = combine(points)

    np.testing.assert_array_equal(outputs["CANDIDATE"][:, :2], expected_geo[:, :2])
    for model in MODEL_IDS:
        np.testing.assert_array_equal(points[model], before[model])
        np.testing.assert_array_equal(outputs[model], before[model])
    assert metadata["fallback_count"] == 0


def test_degenerate_ray_or_radius_falls_back_without_changing_geo():
    geo = np.array(
        [
            [[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0], [EPS, 0.0]],
        ]
    )
    stld = geo.copy()
    stld[:, 2] = [[2.0, 0.0], [0.0, 0.0], [2.0, 0.0]]
    points = {"GEO_A": geo.copy(), "GEO_B": geo.copy()}
    points.update({"STLD1801": stld.copy(), "STLD1901": stld.copy()})
    outputs, metadata = combine(points)

    np.testing.assert_array_equal(outputs["CANDIDATE"], geo)
    np.testing.assert_array_equal(metadata["fallback_mask"], [True, True, True])
    assert metadata["fallback_count"] == 3
    assert metadata["epsilon"] == EPS == 1e-8
