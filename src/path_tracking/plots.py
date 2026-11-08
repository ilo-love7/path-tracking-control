"""绘图：轨迹叠加 / 误差曲线 / 转角曲线 / 指标条形图 / 动画 GIF。"""

from __future__ import annotations

from pathlib import Path as FsPath

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams.update({
    "figure.dpi": 120,
    "font.size": 9,
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

COLORS = {"PurePursuit": "#1f77b4", "Stanley": "#ff7f0e", "MPC": "#2ca02c"}


def _save(fig, path: FsPath) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_trajectory(path, results: dict[str, dict], title: str, out: FsPath) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(path.x, path.y, "k--", lw=1.2, label="参考路径")
    for name, r in results.items():
        f = r["frame"]
        ax.plot(f["x"], f["y"], lw=1.4, label=name, color=COLORS.get(name))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    return _save(fig, out)


def plot_errors(results: dict[str, dict], title: str, out: FsPath) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    for name, r in results.items():
        f = r["frame"]
        axes[0].plot(f["t"], f["e_lat"], label=name, color=COLORS.get(name))
        axes[1].plot(f["t"], f["e_yaw"], label=name, color=COLORS.get(name))
    axes[0].set_ylabel("横向误差 [m]")
    axes[1].set_ylabel("航向误差 [rad]")
    axes[1].set_xlabel("t [s]")
    axes[0].set_title(title)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    return _save(fig, out)


def plot_steer(results: dict[str, dict], title: str, out: FsPath) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    for name, r in results.items():
        f = r["frame"]
        axes[0].plot(f["t"], f["steer"] * 180 / np.pi, label=name, color=COLORS.get(name))
        axes[1].plot(f["t"], f["accel"], label=name, color=COLORS.get(name))
    axes[0].set_ylabel("前轮转角 [deg]")
    axes[1].set_ylabel("加速度 [m/s²]")
    axes[1].set_xlabel("t [s]")
    axes[0].set_title(title)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    return _save(fig, out)


def plot_metric_bars(rows: list[dict], metric_key: str, ylabel: str, title: str, out: FsPath) -> str:
    """rows: [{scenario, model, value}]"""
    scenarios = sorted({r["scenario"] for r in rows})
    models = sorted({r["model"] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(scenarios))
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        vals = [next((r["value"] for r in rows if r["scenario"] == s and r["model"] == m), 0.0)
                for s in scenarios]
        ax.bar(x + i * width, vals, width, label=m, color=COLORS.get(m))
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(scenarios, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, out)


def make_animation(path, frame, title: str, out: FsPath, fps: int = 20, stride: int = 2) -> str:
    """轨迹动画 GIF（降低采样率控制体积）。"""
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(path.x, path.y, "k--", lw=1, label="参考")
    (line,) = ax.plot([], [], "b-", lw=1.5)
    (pt,) = ax.plot([], [], "ro", ms=5)
    ax.set_xlim(path.x.min() - 2, path.x.max() + 2)
    ax.set_ylim(path.y.min() - 2, path.y.max() + 2)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()

    xs = frame["x"].to_numpy()[::stride]
    ys = frame["y"].to_numpy()[::stride]

    def update(i):
        line.set_data(xs[: i + 1], ys[: i + 1])
        pt.set_data([xs[i]], [ys[i]])
        return line, pt

    anim = FuncAnimation(fig, update, frames=len(xs), interval=1000 // fps, blit=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return str(out)
