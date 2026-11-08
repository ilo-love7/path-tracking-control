from __future__ import annotations

import numpy as np

from path_tracking.vehicle import KinematicBicycle, VehicleState


def test_state_defaults():
    s = VehicleState()
    assert (s.x, s.y, s.yaw, s.v) == (0.0, 0.0, 0.0, 0.0)


def test_straight_line_motion(vehicle_cfg):
    veh = KinematicBicycle(**vehicle_cfg)
    veh.reset(0, 0, 0, 10.0)
    for _ in range(10):
        veh.step(accel=0.0, steer=0.0)
    # 1 秒 × 10 m/s ≈ 10 m
    assert abs(veh.state.x - 10.0) < 0.1
    assert abs(veh.state.y) < 1e-6
    assert abs(veh.state.v - 10.0) < 1e-6


def test_constant_steer_turns(vehicle_cfg):
    veh = KinematicBicycle(**vehicle_cfg)
    veh.reset(0, 0, 0, 5.0)
    for _ in range(20):
        veh.step(accel=0.0, steer=0.2)
    assert veh.state.yaw > 0.1
    assert veh.state.y > 0.1


def test_steer_rate_limit(vehicle_cfg):
    veh = KinematicBicycle(**vehicle_cfg)
    veh.reset(0, 0, 0, 5.0)
    veh.step(0.0, 0.5)  # 请求 0.5 rad，但单步只能变 0.5236*0.1=0.0524
    assert abs(veh.prev_steer) <= vehicle_cfg["max_steer_rate"] * vehicle_cfg["dt"] + 1e-9
    assert veh.prev_steer < 0.5


def test_steer_amplitude_limit(vehicle_cfg):
    veh = KinematicBicycle(**vehicle_cfg)
    veh.reset(0, 0, 0, 5.0)
    for _ in range(50):
        veh.step(0.0, 5.0)  # 远超上限
    assert abs(veh.prev_steer) <= vehicle_cfg["max_steer"] + 1e-9


def test_accel_changes_speed(vehicle_cfg):
    veh = KinematicBicycle(**vehicle_cfg)
    veh.reset(0, 0, 0, 5.0)
    for _ in range(10):
        veh.step(accel=1.0, steer=0.0)
    assert veh.state.v > 5.5
