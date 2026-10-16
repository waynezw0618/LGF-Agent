"""lgf_agent.domains.asm.channels — 张量基通道 (T1..T10) 与 23 分量.

背景
----
ASM (代数应力模型) 把 Reynolds 应力各向异性张量展开为张量基:
    b = Σ_m a_m(Θ) · T⁽ᵐ⁾
其中 T⁽ᵐ⁾ 是 S、W (与 Ω) 构成的完整性基张量 (Pope 1975; Gatski-Speziale 1993;
Weatheritt & Sandberg 2016 JCP 325:22-37 eq.(17)-(20))。

本工程的 `dataset_trueasm_*.npz` 给出 T1..T10 的六分量 `c{i}_T{m}` (N,6),
顺序 = [uu, vv, ww, uv, uw, vw], 目标 b23 即第 5 个分量 (INDEX23 = 5, "vw")。

**对称性代数必然 (2026-10-12 诊断)**:
b23 展开中只有 **T1/T5/T6/T10** 四通道的 23 分量为非零, 其余恒零。
且这四条通道的 23 分量与 b23 **同为偶对称** (关于槽道中心 y→-y).

> ⚠️ 历史错误断言: `scripts/rc_asm_vw_t23_full.py:125-126` 曾把
>   "T1s = S₂₃ = ½dW̄/dy 关于槽道中心**反对称**" 写进 physics context。
>   实测 even 得分 ≈ +1.00 (偶对称)。本模块的 `CHANNEL_PARITY` 是**机器可核实**的
>   声明, 由 C1 探针 (domains/asm/constraints.py) 裁决。
"""
from __future__ import annotations

from typing import Dict, List

# 六分量顺序 (与 npz 的 c{i}_T{m} / c{i}_b6 列一一对应)
COMP6: List[str] = ["uu", "vv", "ww", "uv", "uw", "vw"]
INDEX23: int = 5                       # 目标分量在 COMP6 中的下标 → "vw"

# b23 展开中非零通道 (代数必然, 见模块 docstring)
NONZERO_CHANNELS_23: List[int] = [1, 5, 6, 10]

# 通道 23 分量的奇偶性 (关于 y→-y). 全部为 "even" —— 这是被 C1 探针核实的声明。
CHANNEL_PARITY: Dict[int, str] = {m: "even" for m in NONZERO_CHANNELS_23}

# 各通道 23 分量的物理含义 (从既有 FEATURE_ROLES / 物理推导整理)
CHANNEL_DESC: Dict[int, str] = {
    1:  "T1_23 = S_23 = ½ dW̄/dy, 平均剪切 (唯一显式含壁面剪切梯度的通道)",
    5:  "T5_23 ~ S·Ω 型旋转耦合的 23 分量",
    6:  "T6_23 ~ Ω·S 型旋转耦合的 23 分量",
    10: "T10_23 ~ S²·Ω 型高阶耦合的 23 分量",
}

# 白化通道命名 (basisB2 口径)
WHITENED_NAMES: Dict[str, List[str]] = {
    "v": ["T1s", "T5s", "T6s", "T10s"],
    "g": ["T1s", "T5s", "T6s", "T10s"],
    "e": ["T1s", "T5s", "T6s", "T10s"],
}


def channel_legend() -> str:
    lines = [f"- {CHANNEL_DESC[m]}  [奇偶性: 偶对称 (机器核实)]"
             for m in NONZERO_CHANNELS_23]
    return "\n".join(lines)
