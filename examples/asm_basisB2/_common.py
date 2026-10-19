"""示例公共工具: basisB2 数据集解析 (bundled demo → 上游完整数据集)."""
from __future__ import annotations

import os
from typing import Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))          # LGF-Agent/
PARENT = os.path.dirname(REPO)                         # 上层工程 (可能不存在)


def resolve_datasets(explicit_dir: Optional[str] = None) -> Dict[str, str]:
    """按优先级解析 {v,g,e} 三套数据集路径:
    1. 显式目录 (--data, 含 dataset_trueasm_<v>.npz)
    2. 环境变量 LGF_AGENT_ASM_DATA
    3. 本示例打包的精简样例 (64 点/工况, *_demo.npz)
    4. 上层工程完整数据集 outputs/asm/dataset_trueasm_<v>.npz (129 点)
    """
    cands = []
    if explicit_dir:
        cands.append(explicit_dir)
    env = os.environ.get("LGF_AGENT_ASM_DATA")
    if env:
        cands.append(env)
    cands.append(os.path.join(HERE, "data"))
    cands.append(os.path.join(PARENT, "outputs", "asm"))
    for d in cands:
        paths = {v: os.path.join(d, f"dataset_trueasm_{v}.npz") for v in ("v", "g", "e")}
        demo = {v: os.path.join(d, f"dataset_trueasm_{v}_demo.npz") for v in ("v", "g", "e")}
        # 同一目录里混用 demo/完整时按逐个存在优先 demo? 不 — 统一口径:
        # 目录内 *_demo.npz 优先 (示例默认), 否则完整版
        if all(os.path.exists(x) for x in demo.values()):
            return demo
        if all(os.path.exists(x) for x in paths.values()):
            return paths
    raise FileNotFoundError(
        "找不到 v/g/e 三套数据集。请用 --data 指向含 dataset_trueasm_{v,g,e}.npz "
        "的目录 (或运行 pack_demo.py 生成精简样例)。\n已尝试:\n  "
        + "\n  ".join(cands))


def resolve_runs(explicit: Optional[str] = None) -> str:
    """runs_basisB2 目录 (上层工程历史跑批产物; demo 模式下可能不存在)。"""
    for c in ([explicit] if explicit else []) + [
            os.path.join(PARENT, "outputs", "asm", "runs_basisB2")]:
        if c and os.path.isdir(c):
            return c
    return ""


def default_out(name: str) -> str:
    """示例产物目录 (LGF_AGENT_OUTDIR 可覆盖; 默认仓库根 run/)."""
    root = os.environ.get("LGF_AGENT_OUTDIR")
    if not root:
        root = os.path.join(REPO, "run")
    return os.path.join(root, name)
