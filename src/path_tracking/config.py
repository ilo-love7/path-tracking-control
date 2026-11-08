"""配置加载：dataclass + 迷你 YAML（零 PyYAML 依赖，风格对齐 N1–N3）。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def _parse_scalar(tok: str):
    tok = tok.split("#")[0].strip()
    if tok == "":
        return None
    if tok.startswith("[") and tok.endswith("]"):
        inner = tok[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(p.strip()) for p in inner.split(",")]
    if tok.startswith("{") and tok.endswith("}"):
        # inline map {k: v, ...}
        inner = tok[1:-1]
        out = {}
        for part in re.split(r",(?![^\[]*\])", inner):
            if ":" in part:
                k, v = part.split(":", 1)
                out[k.strip()] = _parse_scalar(v.strip())
        return out
    if (tok.startswith('"') and tok.endswith('"')) or (tok.startswith("'") and tok.endswith("'")):
        return tok[1:-1]
    if tok in ("true", "True"):
        return True
    if tok in ("false", "False"):
        return False
    if tok in ("null", "None", "~"):
        return None
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        pass
    return tok


def _load_simple_yaml(text: str) -> dict:
    """解析本项目 config.yaml 用到的子集：嵌套 map / 标量 / 流式 list / 行内 map。

    'key:' 空值后，若下一条缩进更深的行以 '- ' 开头 → list；否则 → map。
    栈帧: (indent, kind, container)  kind ∈ {"map", "list"}
    """
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    root: dict = {}
    stack: list[tuple[int, str, object]] = [(-1, "map", root)]
    i = 0
    while i < len(lines):
        raw = lines[i]
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()
        if line.startswith("- "):
            item = _parse_scalar(line[2:])
            while len(stack) > 1 and indent <= stack[-1][0]:
                stack.pop()
            if stack[-1][1] == "list":
                stack[-1][2].append(item)
            i += 1
            continue
        if ":" not in line:
            i += 1
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        container = stack[-1][2]
        if val.strip() == "":
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            nxt_indent = len(nxt) - len(nxt.lstrip()) if nxt.strip() else 0
            if nxt.strip().startswith("- ") and nxt_indent > indent:
                lst: list = []
                container[key] = lst
                stack.append((indent, "list", lst))
            else:
                child: dict = {}
                container[key] = child
                stack.append((indent, "map", child))
        else:
            container[key] = _parse_scalar(val)
        i += 1
    return root


@dataclass
class VehicleCfg:
    wheelbase: float = 2.9
    max_steer: float = 0.7854
    max_steer_rate: float = 0.5236
    dt: float = 0.1


@dataclass
class MPCParams:
    horizon: int = 5
    q: list[float] = field(default_factory=lambda: [1.0, 1.0, 0.5, 0.5])
    r: list[float] = field(default_factory=lambda: [0.01, 0.01])
    rd: list[float] = field(default_factory=lambda: [0.01, 1.0])
    max_accel: float = 3.0
    max_decel: float = -5.0


@dataclass
class SimCfg:
    init_x: float = 0.0
    init_y: float = 0.0
    init_yaw: float = 0.0
    settle_s: float = 2.0


def load_config(path: str | Path) -> dict:
    """返回嵌套 dict，键与 config.yaml 一致。"""
    text = Path(path).read_text(encoding="utf-8")
    return _load_simple_yaml(text)


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
