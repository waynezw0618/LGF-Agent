"""pytest 夹具: 保证从仓库根导入 lgf_agent, 并提供合成数据与打包样例数据。"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEMO_DATASET = os.path.join(ROOT, "examples", "asm_b23", "data",
                            "dataset_trueasm_g_demo.npz")


@pytest.fixture(scope="session")
def demo_dataset() -> str:
    if not os.path.exists(DEMO_DATASET):
        pytest.skip("打包样例数据缺失 (examples/asm_b23/data/)")
    return DEMO_DATASET


@pytest.fixture()
def synthetic_cases():
    """5 工况、1 自变量; **各工况共用同一函数形式** (仅 x 抽样不同)。

    这样"正确结构 + 全局常数"应当跨工况同等好 —— 才谈得上验证
    cross_case_stability 的 pass 分支。(工况失配的情形在
    test_verify.py::test_cross_case_stability 里用专门构造的数据测。)
    """
    rng = np.random.default_rng(0)
    cases = []
    for ro in (7.5, 15, 30, 75, 150):
        x = rng.uniform(0.05, 1.0, 120)
        y = 0.5 * np.sqrt(x) + 0.3 * x
        cases.append(dict(X=x[:, None], y=y, name=f"Ro{ro:g}"))
    return cases, ["x"]
