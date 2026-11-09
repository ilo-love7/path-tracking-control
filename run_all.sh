#!/usr/bin/env bash
# N4 全流程：00 校验 → 01 对比实验 → 02 动画
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-D:/项目/venvs/flsoh-py312/Scripts/python.exe}"
for s in 00_verify_env 01_comparison 03_deepening 02_animation; do
  echo "======== $s ========"
  "$PY" "scripts/${s}.py"
done
echo "全部完成。结果见 results/ 与 figures/"
