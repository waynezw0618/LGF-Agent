"""lgf_agent.verify.hard_rules — 验收 5 硬规则的泛化 (域无关).

来源: 既有跑批里的"验收 5 硬规则"(坍缩 / 爆参 / 合成作弊 / 跨工况稳定 /
辅助基准异常低)。这里把其中**可域无关化**的 4 条升格为通用探针:

    1. 坍缩        CollapseRule          预测恒定 或 DSL 不含任何变量
    2. 爆参        ParamBlowupRule       max|θ| > blowup_threshold
    3. 跨工况稳定  CrossCaseStabilityRule  per-case NMSE max/min > factor
    4. 值域护栏    RangeGuardRule        预测越界点占比 (V3-c 护栏的探针化)

第 5 条"合成作弊"依赖域定义的**作弊通道/辅助基准**(ASM 域是 T3v/T4v 通道异常低),
故由域适配器实现 (见 domains/asm/constraints.py 的 SynthCheatRule)。
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

from .base import (PASS, VIOLATE, WARN, UNVERIFIED, Constraint, ProbeResult,
                   VerifyContext)


# ---------------------------------------------------------------------------
# 共享工具: 编译并评估候选
# ---------------------------------------------------------------------------
def compile_and_eval(ctx: VerifyContext):
    """返回 (cg, per_case_preds) 或 (None, None)。

    需要的依赖是 core 里的 dsl/compiler —— 只有真正要跑数值探针时才导入,
    保证 store/projector 等纯数据层不背 sympy 的 import 成本。
    """
    if ctx.graph is None:
        from ..core import dsl as _dsl
        try:
            ctx.graph = _dsl.parse_dsl(ctx.dsl, list(ctx.var_names))
        except Exception:
            return None, None
    from ..core.compiler import CompiledGraph
    try:
        cg = CompiledGraph(ctx.graph, backend="numpy")
    except Exception:
        return None, None
    preds: List[np.ndarray] = []
    theta = np.asarray(ctx.theta if ctx.theta is not None else [], dtype=float)
    n_g, n_c = cg.n_g, cg.n_c
    for ci, case in enumerate(ctx.cases or []):
        tl = np.concatenate([theta[:n_g],
                             theta[n_g + ci * n_c: n_g + (ci + 1) * n_c]])
        try:
            preds.append(np.asarray(cg.eval_case(tl, case["X"]), dtype=float))
        except Exception:
            return cg, None
    return cg, preds


def per_case_nmse(preds: Sequence[np.ndarray], cases: Sequence[dict]) -> List[float]:
    out = []
    for p, c in zip(preds, cases):
        y = np.asarray(c["y"], dtype=float)
        w = c.get("w")
        if w is None:
            var = float(np.var(y)) + 1e-12
            mse = float(np.mean((p - y) ** 2))
        else:
            w = np.asarray(w, dtype=float)
            sw = float(w.sum())
            mu = float(np.dot(w, y) / sw)
            var = float(np.dot(w, (y - mu) ** 2) / sw) + 1e-12
            mse = float(np.dot(w, (p - y) ** 2) / sw)
        out.append(mse / var)
    return out


# ---------------------------------------------------------------------------
# 规则 1: 坍缩
# ---------------------------------------------------------------------------
class CollapseRule(Constraint):
    name = "collapse"
    description = "候选坍缩为常数/与输入无关 (预测恒定 或 DSL 无变量)"
    severity = "high"

    def __init__(self, min_rel_std: float = 1e-6):
        self.min_rel_std = min_rel_std

    def check(self, ctx: VerifyContext) -> ProbeResult:
        used = _vars_used(ctx)
        ev: dict = {"vars_used": sorted(used)}
        if not used:
            return ProbeResult(self.name, VIOLATE, ev, self.severity,
                               min_fix="引入至少一个自变量 (梯度/不变量)")
        cg, preds = compile_and_eval(ctx)
        if preds is None:
            return ProbeResult(self.name, UNVERIFIED, {"reason": "eval_failed"},
                               self.severity)
        rels = []
        for p, c in zip(preds, ctx.cases or []):
            y = np.asarray(c["y"], dtype=float)
            s = float(np.std(p)); sy = float(np.std(y)) + 1e-12
            rels.append(s / sy)
        ev["rel_std_per_case"] = [round(r, 8) for r in rels]
        if max(rels) < self.min_rel_std:
            return ProbeResult(self.name, VIOLATE, ev, self.severity,
                               min_fix="公式未随输入变化, 检查是否被常数吸收")
        return ProbeResult(self.name, PASS, ev, self.severity)


# ---------------------------------------------------------------------------
# 规则 2: 爆参
# ---------------------------------------------------------------------------
class ParamBlowupRule(Constraint):
    name = "param_blowup"
    description = "参数幅值爆表 (数值不稳定前兆)"
    severity = "high"

    def __init__(self, threshold: float = 1e3):
        self.threshold = threshold

    def check(self, ctx: VerifyContext) -> ProbeResult:
        th = np.asarray(ctx.theta or [], dtype=float)
        if th.size == 0:
            return ProbeResult(self.name, PASS, {"max_abs_theta": 0.0}, self.severity)
        m = float(np.max(np.abs(th)))
        ev = {"max_abs_theta": m, "threshold": self.threshold,
              "theta": [float(v) for v in th]}
        if not np.isfinite(m) or m > self.threshold:
            return ProbeResult(self.name, VIOLATE, ev, self.severity,
                               min_fix="对参数做尺度归一 / 改写为比值或对数形式")
        return ProbeResult(self.name, PASS, ev, self.severity)


# ---------------------------------------------------------------------------
# 规则 3: 跨工况稳定
# ---------------------------------------------------------------------------
class CrossCaseStabilityRule(Constraint):
    name = "cross_case_stability"
    description = "跨工况泛化: per-case NMSE 的 max/min 比不得超限"
    severity = "high"

    def __init__(self, max_ratio: float = 5.0, min_floor: float = 1e-12):
        self.max_ratio = max_ratio
        self.min_floor = min_floor

    def applies(self, ctx: VerifyContext) -> bool:
        return bool(ctx.cases) and len(ctx.cases) >= 2

    def check(self, ctx: VerifyContext) -> ProbeResult:
        _, preds = compile_and_eval(ctx)
        if preds is None:
            return ProbeResult(self.name, UNVERIFIED, {"reason": "eval_failed"},
                               self.severity)
        nms = per_case_nmse(preds, ctx.cases)
        names = [c.get("name", f"c{i}") for i, c in enumerate(ctx.cases)]
        ev = {"per_case_nmse": {n: round(v, 6) for n, v in zip(names, nms)},
              "max_ratio": self.max_ratio}
        lo = max(min(nms), self.min_floor)
        ratio = float(max(nms) / lo)
        ev["ratio"] = ratio
        if ratio > self.max_ratio:
            worst = names[int(np.argmax(nms))]
            return ProbeResult(self.name, VIOLATE, ev, self.severity,
                               min_fix=f"最差工况 {worst} 主导误差, 需要工况相关的"
                                       f"自变量 (旋转数/雷诺数), 或改结构")
        return ProbeResult(self.name, PASS, ev, self.severity)


# ---------------------------------------------------------------------------
# 规则 4: 值域护栏
# ---------------------------------------------------------------------------
class RangeGuardRule(Constraint):
    name = "range_guard"
    description = "预测越界: |pred| > factor × max|y| 的点占比"
    severity = "medium"

    def __init__(self, factor: float = 20.0, max_frac: float = 0.0):
        self.factor = factor
        self.max_frac = max_frac

    def check(self, ctx: VerifyContext) -> ProbeResult:
        _, preds = compile_and_eval(ctx)
        if preds is None:
            return ProbeResult(self.name, VIOLATE,
                               {"reason": "eval_failed_or_nonfinite"},
                               self.severity,
                               min_fix="存在 NaN/inf 求值, 检查 log/div/exp 定义域")
        worst = 0.0
        per = {}
        for p, c in zip(preds, ctx.cases or []):
            y = np.asarray(c["y"], dtype=float)
            thr = self.factor * (float(np.max(np.abs(y))) + 1e-12)
            frac = float(np.mean(np.abs(p) > thr))
            per[c.get("name", "?")] = frac
            worst = max(worst, frac)
        ev = {"worst_frac": worst, "per_case": per, "factor": self.factor}
        if worst > self.max_frac:
            return ProbeResult(self.name, VIOLATE, ev, self.severity,
                               min_fix="壁面/中心奇点未被抑制 (exp/log/div 定义域)")
        return ProbeResult(self.name, PASS, ev, self.severity)


# ---------------------------------------------------------------------------
# 规则 5: 目标级禁用算子
# ---------------------------------------------------------------------------
class ForbiddenOpRule(Constraint):
    name = "forbidden_ops"
    description = "使用了目标级禁用的算子 (硬拦截的机器化)"
    severity = "high"

    def __init__(self, forbidden: Sequence[str]):
        self.forbidden = [str(x) for x in (forbidden or [])]

    def applies(self, ctx: VerifyContext) -> bool:
        return bool(self.forbidden)

    def check(self, ctx: VerifyContext) -> ProbeResult:
        from ..core.graph import walk
        if ctx.graph is None:
            _, __ = compile_and_eval(ctx)
        if ctx.graph is None:
            return ProbeResult(self.name, UNVERIFIED, {"reason": "parse_failed"},
                               self.severity)
        ops = sorted({n.op for _, n in walk(ctx.graph.root)})
        hit = [o for o in self.forbidden if o in ops]
        ev = {"ops": ops, "forbidden": self.forbidden, "hit": hit}
        if hit:
            return ProbeResult(self.name, VIOLATE, ev, self.severity,
                               min_fix=f"移除算子 {hit} 或改写为等价允许形式")
        return ProbeResult(self.name, PASS, ev, self.severity)


# ---------------------------------------------------------------------------
def _vars_used(ctx: VerifyContext) -> set:
    """候选实际引用的自变量名集合."""
    from ..core.graph import walk
    if ctx.graph is None:
        from ..core import dsl as _dsl
        try:
            ctx.graph = _dsl.parse_dsl(ctx.dsl, list(ctx.var_names))
        except Exception:
            return {v for v in ctx.var_names if v in ctx.dsl}
    used = set()
    for _, n in walk(ctx.graph.root):
        if n.op == "var":
            i = int(n.attrs.get("index", -1))
            if 0 <= i < len(ctx.var_names):
                used.add(ctx.var_names[i])
    return used


def default_hard_rules(*, forbidden: Sequence[str] = (),
                       blowup_threshold: float = 1e3,
                       cross_case_ratio: float = 5.0) -> List[Constraint]:
    """通用硬规则集合 (域无关部分)."""
    return [
        CollapseRule(),
        ParamBlowupRule(blowup_threshold),
        CrossCaseStabilityRule(cross_case_ratio),
        RangeGuardRule(),
        ForbiddenOpRule(forbidden),
    ]
