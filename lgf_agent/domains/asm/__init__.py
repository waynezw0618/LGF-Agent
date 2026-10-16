"""lgf_agent.domains.asm — 旋转槽道流代数应力模型 (ASM) 张量基闭合域.

当前口径: **b23 = g(I₁, I₂, …, Ro_loc, Re_t) · T₂₃** (I_i 为 Pope 完整性基不变量)。

子模块
------
invariants.py   Pope 不变量的语义与白名单 (自变量定义)
channels.py     T1..T10 张量通道 / 23 分量 / 奇偶性 (机器核实)
dataset.py      dataset_trueasm_*.npz 与通用 dataset*.npz 的加载
constraints.py  C1..C9 九条数值探针约束
domain.py       ASMDomain (DomainAdapter 实现)
skills/         领域知识卡 (read_skill 的数据源)
"""
from .domain import ASMDomain                       # noqa: F401
from .constraints import asm_constraints            # noqa: F401
from . import invariants, channels, dataset         # noqa: F401

__all__ = ["ASMDomain", "asm_constraints", "invariants", "channels", "dataset"]
