from __future__ import annotations

import numpy as np

from path_tracking.metrics import cross_track_error, heading_error, metric_pack


def test_cross_track_zero_on_path():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0])
    yaw = np.zeros(3)
    e = cross_track_error(x, y, yaw, x, y, yaw)
    assert np.allclose(e, 0.0)


def test_cross_track_sign_left():
    # 车在参考上方（左侧）→ 正
    x = np.array([0.0])
    y = np.array([1.0])
    yaw = np.array([0.0])
    e = cross_track_error(x, y, yaw, np.array([0.0]), np.array([0.0]), np.array([0.0]))
    assert e[0] > 0.5


def test_heading_error_wrap():
    e = heading_error(np.array([np.pi * 0.9]), np.array([-np.pi * 0.9]))
    assert abs(abs(e[0]) - np.pi * 0.2) < 1e-9  # wraps to -0.2pi or 0.2pi
    assert abs(e[0]) < np.pi


def test_metric_pack_keys():
    n = 50
    m = metric_pack(np.zeros(n), np.zeros(n), 0.1,
                    np.zeros(n), np.zeros(n), solve_times=np.full(n, 0.002))
    assert set(m) >= {"lat_rms", "lat_max", "yaw_rms", "steer_rate_rms",
                      "steer_jerk_rms", "accel_rms", "n_steps", "solve_ms_mean"}
    assert m["lat_rms"] == 0.0
    assert m["n_steps"] == n
    assert abs(m["solve_ms_mean"] - 2.0) < 1e-6


def test_metric_pack_steer_rate():
    n = 10
    steer = np.linspace(0, 0.5, n)  # 变化 0.5/0.9s
    m = metric_pack(steer, np.zeros(n), 0.1, np.zeros(n), np.zeros(n))
    assert m["steer_rate_rms"] > 0.5
