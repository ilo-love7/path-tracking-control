from __future__ import annotations

import numpy as np

from path_tracking.path import (
    CubicSpline2D, build_path, make_circle, make_double_lane_change,
    nearest_index,
)


def test_straight_path_geometry(simple_path):
    p = simple_path
    assert len(p) == 200
    assert np.allclose(p.yaw, 0.0, atol=1e-6)
    assert np.allclose(p.curvature, 0.0, atol=1e-3)


def test_circle_geometry(circle_path):
    p = circle_path
    # 曲率 ≈ 1/30
    mid = slice(10, -10)
    assert np.allclose(p.curvature[mid], 1.0 / 30.0, atol=1e-3)
    # 起点在 (0, 0)
    assert abs(p.x[0]) < 1e-6 and abs(p.y[0]) < 1e-6
    # 弧长单调
    assert np.all(np.diff(p.s) > 0)


def test_dlc_shape(dlc_path):
    p = dlc_path
    assert p.y.max() > 3.0  # 横移 3.5 m
    assert p.y[-1] < 0.5    # 回到 0
    assert np.all(np.diff(p.s) > 0)


def test_cubic_spline_interpolation():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    sp = CubicSpline2D(x, y)
    # 端点精确
    assert abs(sp.position(0.0)[0] - 0.0) < 1e-9
    assert abs(sp.position(0.0)[1] - 0.0) < 1e-9


def test_nearest_index_local(simple_path):
    i = nearest_index(simple_path, 50.0, 0.1, hint=100, window=50)
    assert abs(simple_path.x[i] - 50.0) < 1.0


def test_build_path_factory():
    cfg = {"segments": [{"x": 0, "y": 0}, {"x": 50, "y": 0}, {"x": 80, "y": 3.5}, {"x": 130, "y": 0}]}
    p = build_path("double_lane_change", cfg)
    assert len(p) > 100
    p2 = build_path("circle", {"radius": 30.0, "arc_deg": 270.0, "center_x": 0.0, "center_y": 30.0})
    assert len(p2) > 100


def test_clothoid_s_curve_shape():
    from path_tracking.path import make_clothoid_s_curve
    p = make_clothoid_s_curve(kappa_max=0.05, yaw_up=0.5, yaw_down=0.7, straight_len=15.0)
    assert len(p) > 100
    # 峰值曲率应接近 kappa_max
    assert abs(np.max(np.abs(p.curvature)) - 0.05) < 0.01
    # S 弯：航向先正后负，y 穿越两侧
    assert np.max(p.yaw) > 0.2
    assert np.min(p.yaw) < -0.2
    assert np.max(p.y) > 1.0
    assert np.min(p.y) < -1.0


def test_clothoid_in_build_path():
    p = build_path("clothoid_s", {"kappa_max": 0.05, "yaw_up": 0.5,
                                  "yaw_down": 0.7, "straight_len": 15.0})
    assert len(p) > 50
    assert p.s[-1] > 80.0
