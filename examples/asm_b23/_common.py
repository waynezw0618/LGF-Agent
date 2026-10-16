"""示例公共工具: 数据集路径解析 (bundled demo → 上游完整数据集)."""
from __future__ import annotations

import os
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))          # LGF-Agent/
PARENT = os.path.dirname(REPO)                         # 上层工程目录 (可能不存在)


def resolve_dataset(explicit: Optional[str] = None, variant: str = "g") -> str:
    """按优先级查找数据集:
    1. 显式传入 (--dataset)
    2. 环境变量 LGF_AGENT_DATASET
    3. 本示例打包的精简样例 (64 点/工况, 建议用于冒烟与 CI)
    4. 上层工程的完整数据集 outputs/asm/dataset_trueasm_<variant>.npz
    """
    cands = []
    if explicit:
        cands.append(explicit)
    env = os.environ.get("LGF_AGENT_DATASET")
    if env:
        cands.append(env)
    cands.append(os.path.join(HERE, "data", f"dataset_trueasm_{variant}_demo.npz"))
    cands.append(os.path.join(HERE, "data", f"dataset_trueasm_{variant}.npz"))
    cands.append(os.path.join(PARENT, "outputs", "asm",
                              f"dataset_trueasm_{variant}.npz"))
    for c in cands:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "找不到数据集。请用 --dataset 指定 dataset_trueasm_<variant>.npz。\n已尝试:\n  "
        + "\n  ".join(cands))
