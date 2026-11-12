"""filtering.py 单测：EMA 滤波 + 最近点滞回。"""
import numpy as np
import pytest

from path_tracking.filtering import HysteresisNearest, StateFilter


class TestStateFilter:
    def test_alpha_validation(self):
        with pytest.raises(ValueError):
            StateFilter(alpha=0.0)
        with pytest.raises(ValueError):
            StateFilter(alpha=1.5)

    def test_first_update_passes_through(self):
        f = StateFilter(alpha=0.3)
        z = np.array([1.0, 2.0, 0.5, 10.0])
        out = f.update(z)
        assert np.array_equal(out, z)

    def test_ema_converges_to_constant(self):
        f = StateFilter(alpha=0.3)
        z = np.array([5.0, 5.0, 0.0, 5.0])
        f.update(z)
        for _ in range(200):
            out = f.update(z)
        assert np.allclose(out, z, atol=1e-6)

    def test_ema_smooths_noise(self):
        rng = np.random.default_rng(42)
        f = StateFilter(alpha=0.2)
        true_val = 10.0
        f.update(np.array([true_val, true_val, 0.0, true_val]))
        outputs = []
        for _ in range(500):
            noisy = np.array([true_val + rng.normal(0, 1.0),
                              true_val + rng.normal(0, 1.0),
                              0.0,
                              true_val + rng.normal(0, 1.0)])
            outputs.append(f.update(noisy))
        out_arr = np.array(outputs)
        raw_std = 1.0
        filt_std = out_arr[:, 0].std()
        assert filt_std < raw_std * 0.5  # 至少压一半

    def test_yaw_wraparound(self):
        f = StateFilter(alpha=0.5, n_channels=4)
        f.update(np.array([0.0, 0.0, 3.0, 0.0]))  # yaw ≈ 3.0
        out = f.update(np.array([0.0, 0.0, -3.0, 0.0]))  # yaw 跳到 −3.0（环绕差 ≈ 0.28）
        # 正确环绕：yaw 应在 3.0 和 −3.0 之间平滑过渡（≈ π 或 −π 附近）
        assert abs(out[2]) > 2.5  # 仍在 |yaw| 接近 π 的区域

    def test_dimension_mismatch(self):
        f = StateFilter(alpha=0.3, n_channels=4)
        with pytest.raises(ValueError):
            f.update(np.array([1.0, 2.0]))

    def test_reset(self):
        f = StateFilter(alpha=0.3)
        f.update(np.array([1.0, 1.0, 0.0, 1.0]))
        f.reset()
        z = np.array([9.0, 9.0, 0.0, 9.0])
        out = f.update(z)
        assert np.array_equal(out, z)  # reset 后首值直通


class _FakePath:
    """简化路径对象（只给 x/y）。"""
    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)


class TestHysteresisNearest:
    def _line_path(self, n=100):
        return _FakePath(np.linspace(0, 50, n), np.zeros(n))

    def test_forward_always_accepted(self):
        h = HysteresisNearest(window=10, hysteresis=3)
        path = self._line_path()
        h.reset(0)
        # 逐步前进
        for i in [5, 10, 15, 20]:
            idx = h.update(path, path.x[i], 0.0)
        assert idx == 20

    def test_small_backward_rejected(self):
        h = HysteresisNearest(window=10, hysteresis=5)
        path = self._line_path()
        h.reset(50)
        # 扰动到 48（回退 2 < hysteresis 5）→ 维持 50
        idx = h.update(path, path.x[48], 0.1)
        assert idx == 50

    def test_large_backward_accepted(self):
        h = HysteresisNearest(window=30, hysteresis=3)
        path = self._line_path()
        h.reset(50)
        # 回退到 40（回退 10 > hysteresis 3）→ 接受
        idx = h.update(path, path.x[40], 0.0)
        assert idx == 40

    def test_window_limits_search(self):
        h = HysteresisNearest(window=5, hysteresis=0)
        path = self._line_path()
        h.reset(50)
        # 目标在 idx=80（超窗）→ 只能在 [45, 55] 找到 55
        idx = h.update(path, path.x[80], 0.0)
        assert idx == 55

    def test_validation(self):
        with pytest.raises(ValueError):
            HysteresisNearest(window=0)
        with pytest.raises(ValueError):
            HysteresisNearest(hysteresis=-1)

    def test_noise_robustness(self):
        """噪声下滞回索引比裸 nearest_index 稳定得多。"""
        rng = np.random.default_rng(7)
        n = 200
        path = _FakePath(np.linspace(0, 100, n), np.zeros(n))
        h = HysteresisNearest(window=15, hysteresis=3)
        h.reset(0)
        noisy_indices = []
        true_indices = []
        for step in range(100):
            true_i = step * 2  # 匀速前进
            if true_i >= n:
                break
            nx = path.x[true_i] + rng.normal(0, 0.5)  # σ=0.5 m（对应 E9 的 σ=50 cm 量级）
            ny = 0.0 + rng.normal(0, 0.5)
            idx = h.update(path, nx, ny)
            noisy_indices.append(idx)
            true_indices.append(true_i)
        # 滞回索引与真值的平均偏差应远小于窗口
        err = np.mean(np.abs(np.array(noisy_indices) - np.array(true_indices)))
        assert err < 5.0
