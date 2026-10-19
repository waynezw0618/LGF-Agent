"""lgf_agent.domains.asm — 旋转槽道流代数应力模型 (ASM) 张量基闭合域.

两套口径并存 (同一内核, 同一 13 动作):

1. **单通道口径** (旧, asm_b23 示例): b23 = g(I₁,…,Ro_loc,Re_t)·T₂₃
   — dataset.py / domain.py (ASMDomain), C1..C9 探针。
2. **多通道系数口径** (新, 2026-10-18/19 报告轮, asm_basisB2 示例):
   b23 = Σ_j s_j·a_j(Θ)·T̃_j23 — basisb2.py (BasisB2Domain, 白化/靶标/两阶段)
   + refit.py (终口径联合重拟合; 可独立复算报告冠军 0.0576)。

子模块
------
invariants.py   Pope 不变量的语义与白名单 (自变量定义)
channels.py     T1..T10 张量通道 / 23 分量 / 奇偶性 (机器核实)
dataset.py      dataset_trueasm_*.npz 与通用 dataset*.npz 的加载 (单通道口径)
constraints.py  C1..C9 九条数值探针约束 (两口径共用)
domain.py       ASMDomain (单通道口径 DomainAdapter)
basisb2.py      多通道系数口径: load_channels / stage_base / BasisB2Domain
refit.py        终口径: compile_expr / joint_refit / 历史跑批回放
skills/         领域知识卡 (read_skill 的数据源)
"""
from .domain import ASMDomain                       # noqa: F401
from .constraints import asm_constraints            # noqa: F401
from . import invariants, channels, dataset         # noqa: F401
from . import basisb2, refit                        # noqa: F401
from .basisb2 import BasisB2Domain                  # noqa: F401

__all__ = ["ASMDomain", "BasisB2Domain", "asm_constraints",
           "invariants", "channels", "dataset", "basisb2", "refit"]
