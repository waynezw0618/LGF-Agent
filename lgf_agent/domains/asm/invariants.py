"""lgf_agent.domains.asm.invariants — Pope 不变量 (ASM 域的自变量语义).

S. B. Pope, *A more general effective-viscosity hypothesis*, JFM 72 (1975) 331-340。

对对称张量 S 与反对称张量 W, 张量多项式的**完整性基** (integrity basis) 为 5 个
标量不变量 (本工程的数据集 `dataset_trueasm_*.npz` 的 `c{i}_scalars` 列):

    列 0  eta2 = tr(S²)      Pope I₁
    列 1  zeta2 = tr(W²)     Pope I₂
    列 2  s3    = tr(S³)     Pope I₃
    列 3  so2   = tr(S·W²)   Pope I₄
    列 4  s2o2  = tr(S²:W²)  Pope I₅
    列 5  ret   = Re_t = k²/(νε)   局部湍流雷诺数 (局部分辨标量扩展)
    列 6  roloc = Ro_loc = 2Ωk/ε   局部旋转数 (区分旋转工况的核心变量)

注意区分两种 W 定义 (影响 zeta2/so2/s2o2 的数值, 不影响本模块的结构):
- 变体 g: Gatski-Speziale 1993, W* = W̄ + 1·Ω
- 变体 e: 纯 W̄; 变体 v: 其他归一
三者在 `dataset_trueasm_{g,e,v}.npz` 中分别给出, `variant` 字段记录。

**为什么单靠不变量不够 (本工程 2026-10-11 结论)**:
vw-only 轮加真壁距 yw 后 loss 才到 SOTA —— 局部不变量不足以决定近壁剖面,
壁面信息必要。故 Θ 中额外含 yw = 1-|y/h| ∈ [0,1]。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

# 数据集中 c{i}_scalars 的列语义 (顺序不可改: 与既有 npz 绑定)
SCALAR_COLUMNS: List[str] = ["eta2", "zeta2", "s3", "so2", "s2o2", "ret", "roloc"]

# Pope 完整性基的 5 个不变量 (供 prompt / 约束层引用)
POPE_INVARIANTS: Dict[str, Dict[str, str]] = {
    "eta2":  {"pope": "I1", "expr": "tr(S^2)",   "desc": "Pope I1, 应变率强度 (≥0)"},
    "zeta2": {"pope": "I2", "expr": "tr(W^2)",   "desc": "Pope I2, 旋转率强度 (≥0)"},
    "s3":    {"pope": "I3", "expr": "tr(S^3)",   "desc": "Pope I3, 应变率三阶不变量"},
    "so2":   {"pope": "I4", "expr": "tr(S W^2)", "desc": "Pope I4, 应变-旋转耦合"},
    "s2o2":  {"pope": "I5", "expr": "tr(S^2 W^2)", "desc": "Pope I5, 应变²-旋转²耦合"},
}

# 局部标量扩展 (非 Reynolds 不变量, 但为允许的合法局部量)
LOCAL_EXTENSIONS: Dict[str, Dict[str, str]] = {
    "ret":   {"expr": "k^2/(nu eps)", "desc": "局部湍流雷诺数 Re_t (低-Re 近壁阻尼变量)"},
    "roloc": {"expr": "2 Omega k/eps", "desc": "局部旋转数 Ro_loc (带符号, 区分旋转工况)"},
}

# 壁面扩展 (非局部量, 但本工程已证明必要)
WALL_EXTENSIONS: Dict[str, Dict[str, str]] = {
    "yw": {"expr": "1-|y/h|", "desc": "无量纲真壁距, 0=壁面 1=中心"},
}

# 默认自变量集合: Pope I1..I5 + Re_t + Ro_loc (+ yw)
DEFAULT_VARS: List[str] = ["eta2", "zeta2", "s3", "so2", "s2o2", "ret", "roloc", "yw"]

# 在 g(I)·T 口径下强烈推荐的最小集合 (过大会显著扩大搜索空间)
MINIMAL_VARS: List[str] = ["eta2", "zeta2", "s2o2", "ret", "roloc", "yw"]


def invariant_legend() -> str:
    """生成 prompt 用的不变量图例 (与 PHYS 文本一致, 单一来源)。"""
    lines = []
    for name, meta in POPE_INVARIANTS.items():
        lines.append(f"- {name} = {meta['expr']}  ({meta['pope']}); {meta['desc']}")
    for name, meta in LOCAL_EXTENSIONS.items():
        lines.append(f"- {name} = {meta['expr']}; {meta['desc']}")
    for name, meta in WALL_EXTENSIONS.items():
        lines.append(f"- {name} = {meta['expr']}; {meta['desc']} (非局部量, 允许的壁面扩展)")
    return "\n".join(lines)


def variables_for(mode: str = "minimal") -> List[str]:
    if mode == "full":
        return list(DEFAULT_VARS)
    if mode == "pope_only":
        return [v for v in MINIMAL_VARS if v != "yw"]
    return list(MINIMAL_VARS)
