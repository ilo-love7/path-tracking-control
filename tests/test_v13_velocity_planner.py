"""velocity_planner.py 单测：安全速度剖面与加减速约束。"""
import numpy as np
import pytest

from path_tracking.velocity_planner import noise_decay, plan_speed_profile, safe_speed


class TestNoiseDecay:
    def test_zero_noise(self):
        assert noise_decay(0.0) == pytest.approx(1.0)

    def test_monotone_decreasing(self):
        sigmas = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
        vals = [noise_decay(s) for s in sigmas]
        for i in range(1, len(vals)):
            assert vals[i] < vals[i - 1]

    def test_large_noise_near_zero(self):
        assert noise_decay(0.10) < 0.01
        assert noise_decay(0.20) < 1e-6


class TestSafeSpeed:
    def test_no_noise_limited_by_curve(self):
        v = safe_speed(30.0, sigma_m=0.0)
        assert v == pytest.approx(np.sqrt(0.35 * 9.81 * 30.0), rel=1e-6)

    def test_noise_reduces_speed(self):
        v0 = safe_speed(30.0, sigma_m=0.0)
        v5 = safe_speed(30.0, sigma_m=0.05)
        assert v5 < v0

    def test_vmax_cap(self):
        v = safe_speed(1e6, sigma_m=0.0, v_max=16.667)
        assert v == pytest.approx(16.667)

    def test_vmin_floor(self):
        v = safe_speed(1.0, sigma_m=0.20, v_min=1.0)
        assert v == pytest.approx(1.0)

    def test_invalid_radius(self):
        with pytest.raises(ValueError):
            safe_speed(0.0)
        with pytest.raises(ValueError):
            safe_speed(-5.0)


class TestPlanSpeedProfile:
    def test_straight_line_constant(self):
        kappa = np.zeros(100)
        v = plan_speed_profile(kappa, sigma_m=0.0)
        assert v.shape == (100,)
        assert np.all(v == pytest.approx(16.667, rel=1e-6))

    def test_tight_curve_slows_down(self):
        kappa = np.full(50, 0.05)  # R = 20 m
        v = plan_speed_profile(kappa, sigma_m=0.0)
        v_expected = safe_speed(20.0, 0.0)
        assert v[25] == pytest.approx(min(v_expected, 16.667), rel=1e-6)

    def test_accel_decel_smoothness(self):
        # 前半直路（高速）后半急弯（低速）→ 减速平滑
        n = 200
        kappa = np.zeros(n)
        kappa[100:] = 0.1  # R = 10 m
        v = plan_speed_profile(kappa, sigma_m=0.0, a_max=2.0, ds=0.5)
        dv = np.diff(v)
        # 每步速度变化不超过 a_max * dt ≈ a_max * ds / v（近似）
        # 宽松检验：相邻速度差不超过 2.0（已由 a_max 平滑约束）
        assert np.max(np.abs(dv)) < 2.0

    def test_noise_lowers_entire_profile(self):
        kappa = np.full(50, 0.02)
        v0 = plan_speed_profile(kappa, sigma_m=0.0)
        v5 = plan_speed_profile(kappa, sigma_m=0.05)
        assert np.all(v5 <= v0 + 1e-9)
        assert v5.mean() < v0.mean()

    def test_zero_curvature_full_speed(self):
        kappa = np.array([0.0, 0.0, 0.0])
        v = plan_speed_profile(kappa, sigma_m=0.0, v_max=20.0)
        assert np.all(v == pytest.approx(20.0))
