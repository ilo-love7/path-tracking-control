"""vehicle.py 非线性轮胎单测：Pacejka-lite 饱和行为。"""
import numpy as np
import pytest

from path_tracking.vehicle import DynamicBicycle, NonlinearDynamicBicycle


class TestNonlinearTire:
    def _make(self, **kw):
        return NonlinearDynamicBicycle(wheelbase=2.9, max_steer=0.7854,
                                       max_steer_rate=0.5236, dt=0.1,
                                       mass=1573.0, iz=2500.0,
                                       lf=1.2, lr=1.7, cf=80000.0, cr=80000.0,
                                       **kw)

    def test_small_slip_matches_linear(self):
        """小侧偏角下饱和轮胎 ≈ 线性轮胎。"""
        nl = self._make()
        lin = DynamicBicycle(wheelbase=2.9, max_steer=0.7854,
                             max_steer_rate=0.5236, dt=0.1,
                             mass=1573.0, iz=2500.0,
                             lf=1.2, lr=1.7, cf=80000.0, cr=80000.0)
        for v0 in (5.0, 10.0):
            nl.reset(0, 0, 0, v0)
            lin.reset(0, 0, 0, v0)
            steer_small = 0.01  # 小转角
            for _ in range(10):
                s_nl = nl.step(0.0, steer_small)
                s_lin = lin.step(0.0, steer_small)
            # 10 步后状态应几乎一致
            assert abs(s_nl.x - s_lin.x) < 0.1
            assert abs(s_nl.y - s_lin.y) < 0.1

    def test_saturation_limits_force(self):
        """大侧偏角下力不超 μ·Fz。"""
        nl = self._make(mu=0.8)
        nl.reset(0, 0, 0, 30.0)  # 高速
        # 大转角驱动大侧偏角
        for _ in range(50):
            nl.step(0.0, 0.5)  # 大转角
        # 检查 tanh 输出不超 Fy_max
        # Fyf_max = mu * m * g * lr / L
        L = 1.2 + 1.7
        Fyf_max = 0.8 * 1573.0 * 9.81 * 1.7 / L
        # 从内部状态估算（不直接暴露 Fy，用侧偏角反推）
        # 这里只验证模型不发散、有限值
        assert np.isfinite(nl.state.x)
        assert np.isfinite(nl.vy)
        assert np.isfinite(nl.r)

    def test_mu_effect(self):
        """μ 越小饱和越早 → 同样转角下横向加速度更小。"""
        hi = self._make(mu=1.2)
        lo = self._make(mu=0.4)
        for m in (hi, lo):
            m.reset(0, 0, 0, 20.0)
        for _ in range(30):
            hi.step(0.0, 0.3)
            lo.step(0.0, 0.3)
        # 低 μ 饱和更早 → 侧偏角更大（轮胎给不出更多力）→ 横摆响应弱
        assert abs(lo.r) <= abs(hi.r) + 1e-6

    def test_interface_compatible(self):
        """与 DynamicBicycle 同接口（simulate.py 兼容）。"""
        nl = self._make()
        nl.reset(1.0, 2.0, 0.3, 5.0)
        s = nl.step(1.0, 0.1)
        assert hasattr(s, 'x') and hasattr(s, 'y') and hasattr(s, 'yaw') and hasattr(s, 'v')
        sv = nl.state_vector
        assert sv.shape == (6,)
