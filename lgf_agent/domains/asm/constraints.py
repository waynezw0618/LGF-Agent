"""lgf_agent.domains.asm.constraints — ASM 域 9 条约束 (全部数值探针).

| # | 约束 | 探针做的事 | 对应既有资产 |
|---|------|-----------|-------------|
| C1 | 奇偶对称性   | 对 y→−y 镜像后算 even 得分 | **今天 T1s 错误断言的机器化** |
| C2 | 张量形式可表性 | 在 T 基上投影残差 + 基共线性 cond | `_tmp_t23_basis_diag*.py` (cond 136–234) |
| C3 | 值域         | b23 符号一致性 / 量级带 | `attribution._target_semantics` |
| C4 | 跨工况常数共享 | 是否存在逐工况常数 K# | `_force_global_params` |
| C5 | 跨 Ro 泛化    | LOSO mean / worst-case | 既有 LOSO 脚本 (标量 0.80 / T 基爆) |
| C6 | 量纲一致性    | 自变量是否全为已知无量纲量 | `priors.dimensional_analysis` |
| C7 | 数值鲁棒性    | yw→0 / 中心 / 外推抽样, NaN·inf·负 log | `paramfit._count_finite` |
| C8 | 复杂度        | 节点/参数上限 | `cfg.max_edges / max_params` |
| C9 | 反作弊        | 在诱饵目标上的拟合是否异常好 | 验收 5 硬规则中"合成作弊" |

设计原则: 每条约束都必须给出**可复现的数值证据** (Tacke et al. 2026 的数值探针法),
而不是文本断言。无法数值化的物理直觉一律 `unverified`, 不得作为约束使用。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ...verify.base import (PASS, VIOLATE, WARN, UNVERIFIED, Constraint,
                            ProbeResult, VerifyContext)
from ...verify.hard_rules import compile_and_eval, per_case_nmse


# ---------------------------------------------------------------------------
def even_score(yh: np.ndarray, values: np.ndarray, nb: int = 12) -> float:
    """偶对称得分 ∈ [-1, 1]: 对 y→-y 镜像后按 |yh| 分箱比较左右两侧。

    ≈ +1 → 偶对称; ≈ -1 → 奇对称; ≈ 0 → 不对称。
    这是 C1 探针与"断言审计"共用的核心度量 (单一来源, 避免两处实现漂移)。
    """
    yh = np.asarray(yh, dtype=float)
    v = np.asarray(values, dtype=float)
    if len(yh) < 8 or not np.all(np.isfinite(v)):
        return 0.0
    ay = np.abs(yh)
    qs = np.linspace(0.0, 1.0, int(nb) + 1) * (ay.max() + 1e-12)
    left, right = [], []
    for b in range(int(nb)):
        ml = (ay >= qs[b]) & (ay <= qs[b + 1]) & (yh < 0)
        mr = (ay >= qs[b]) & (ay <= qs[b + 1]) & (yh > 0)
        if ml.sum() and mr.sum():
            left.append(float(np.mean(v[ml])))
            right.append(float(np.mean(v[mr])))
    if len(left) < 3:
        return 0.0
    a, b_ = np.asarray(left), np.asarray(right)
    # 归一化到 [-1, 1]: +1 = 完全偶 (a==b); -1 = 完全奇 (a==-b)
    num = float(np.mean((a + b_) ** 2) - np.mean((a - b_) ** 2))
    den = float(np.mean((a + b_) ** 2) + np.mean((a - b_) ** 2)) + 1e-12
    return num / den


def _extras(ctx: VerifyContext) -> Dict[str, Any]:
    e = getattr(ctx, "extras", None) or {}
    if e:
        return e
    d = ctx.domain
    if d is not None and hasattr(d, "extras_cache"):
        return d.extras_cache
    return {}


def _preds(ctx: VerifyContext):
    return compile_and_eval(ctx)


# ===========================================================================
# C1 奇偶对称性
# ===========================================================================
class ParityRule(Constraint):
    name = "C1_parity"
    description = "候选作为 yh 的函数须为偶对称 (与 b23 同); 对 y→-y 镜像后比较"
    severity = "high"

    def __init__(self, tol: float = 0.15, yw_var: str = "yw"):
        self.tol = tol
        self.yw_var = yw_var

    def applies(self, ctx: VerifyContext) -> bool:
        ex = _extras(ctx)
        return bool(ex.get("case_extras"))

    def check(self, ctx: VerifyContext) -> ProbeResult:
        ex = _extras(ctx)
        ces = ex["case_extras"]
        if self.yw_var not in list(ctx.var_names):
            return ProbeResult(self.name, UNVERIFIED,
                               {"reason": "no_wall_variable"}, self.severity,
                               note="缺少 yw, 无法构造镜像")
        _, preds = _preds(ctx)
        if preds is None:
            return ProbeResult(self.name, UNVERIFIED, {"reason": "eval_failed"},
                               self.severity)
        scores, per = [], {}
        for p, e in zip(preds, ces):
            sc = even_score(e.yh, p)
            scores.append(sc)
            per[f"Ro{e.ro:g}"] = round(sc, 4)
        mean_sc = float(np.mean(scores)) if scores else 0.0
        ev = {"even_score_mean": round(mean_sc, 4), "per_case": per,
              "tol": self.tol,
              "reference": "b23 与 T1/T5/T6/T10_23 通道均为偶对称 (even≈+1.00)"}
        if mean_sc < self.tol:
            return ProbeResult(self.name, VIOLATE, ev, self.severity,
                               min_fix="预测随 yh 呈奇/非对称 → 改用 |yh| 或 yw 的偶函数 "
                                       "(如 yw、yw²、log(yw)) 替换 yh 的奇次项")
        if mean_sc < 0.6:
            return ProbeResult(self.name, WARN, ev, self.severity,
                               min_fix="偶对称性略弱, 检查是否含 yh 的一次项")
        return ProbeResult(self.name, PASS, ev, self.severity)


# ===========================================================================
# C2 张量形式可表性
# ===========================================================================
class TensorRepresentabilityRule(Constraint):
    name = "C2_tensor_basis"
    description = "b23 能否写成 Σ a_m·T_m23: T 基投影残差 + 基共线性 cond"
    severity = "medium"

    def applies(self, ctx: VerifyContext) -> bool:
        ex = _extras(ctx)
        return bool(ex.get("case_extras")) and ex["case_extras"][0].channels is not None

    def check(self, ctx: VerifyContext) -> ProbeResult:
        ex = _extras(ctx)
        ces = ex["case_extras"]
        Zs, ys, ws = [], [], []
        for e in ces:
            if e.channels is None:
                continue
            Zs.append(e.channels)
            ys.append(e.b23)
            ws.append(np.full(len(e.b23), 1.0 / (len(e.b23) * (e.var_c + 1e-300))))
        if not Zs:
            return ProbeResult(self.name, UNVERIFIED, {"reason": "no_channels"},
                               self.severity)
        Z = np.vstack(Zs); y = np.concatenate(ys); w = np.concatenate(ws)
        sw = np.sqrt(w)
        Zn = Z * sw[:, None]; yn = y * sw
        sc = np.sqrt((Zn ** 2).sum(0)); sc[sc <= 0] = 1.0
        Zn = Zn / sc
        G = Zn.T @ Zn
        cond = float(np.linalg.cond(G))
        coef, *_ = np.linalg.lstsq(Zn, yn, rcond=1e-11)
        resid = yn - Zn @ coef
        proj_res = float(np.mean(resid ** 2) / (np.mean(yn ** 2) + 1e-300))
        ev = {"cond": cond, "projection_residual": proj_res,
              "n_channels": int(Z.shape[1]),
              "cancel_ratio": _cancel_ratio(Z, coef, y, w)}
        if cond > 1e6 or proj_res > 0.5:
            return ProbeResult(self.name, WARN, ev, self.severity,
                               min_fix="T 基近共线/不可表 → 换白化正交基 (basisB2 口径), "
                                       "或让同一组系数同时闭合 6 分量")
        return ProbeResult(self.name, PASS, ev, self.severity)


def _cancel_ratio(Z: np.ndarray, coef: np.ndarray, y: np.ndarray,
                  w: np.ndarray) -> float:
    """抵消比 Σ‖C_m‖/‖b23‖ —— 病态性的直接量度 (2026-10-16: 99–32201×)。"""
    sw = np.sqrt(w)
    terms = (Z * sw[:, None]) * coef[None, :]
    num = float(np.sum(np.abs(terms), axis=1).mean())
    den = float(np.abs((Z @ coef) * sw).mean()) + 1e-300
    return num / den


# ===========================================================================
# C3 值域
# ===========================================================================
class RangeRule(Constraint):
    name = "C3_range"
    description = "b23 恒正且偏置主导 —— 检查预测的符号一致性与量级带"
    severity = "medium"

    def applies(self, ctx: VerifyContext) -> bool:
        ex = _extras(ctx)
        return bool(ex.get("case_extras")) and ctx.cases is not None

    def check(self, ctx: VerifyContext) -> ProbeResult:
        _, preds = _preds(ctx)
        if preds is None:
            return ProbeResult(self.name, UNVERIFIED, {"reason": "eval_failed"},
                               self.severity)
        agree, ratios = [], []
        for p, c in zip(preds, ctx.cases):
            y = np.asarray(c["y"], dtype=float)
            p = np.asarray(p, dtype=float)
            m = np.abs(y) > 1e-12
            if m.sum():
                agree.append(float(np.mean(np.sign(p[m]) == np.sign(y[m]))))
            ratios.append(float((np.mean(np.abs(p)) + 1e-300)
                                / (np.mean(np.abs(y)) + 1e-300)))
        ev = {"sign_agreement": round(float(np.mean(agree)) if agree else 0.0, 4),
              "mag_ratio_mean": round(float(np.mean(ratios)), 4),
              "note": "真值 b23 全 y 恒正、偶对称单峰、偏置主导 (mean/std 1.55-2.91)"}
        sa = ev["sign_agreement"]
        if sa < 0.75:
            return ProbeResult(self.name, VIOLATE, ev, self.severity,
                               min_fix="预测符号与 b23 大面积不符 → 需要常数偏置项 "
                                       "(C#) 或正定结构 (exp/平方/比值)")
        if sa < 0.95 or not (0.2 <= ev["mag_ratio_mean"] <= 5.0):
            return ProbeResult(self.name, WARN, ev, self.severity,
                               min_fix="量级带偏离, 检查是否被个别工况主导")
        return ProbeResult(self.name, PASS, ev, self.severity)


# ===========================================================================
# C4 跨工况常数共享
# ===========================================================================
class GlobalParamsOnlyRule(Constraint):
    name = "C4_global_params"
    description = "ASM 物理要求 5 工况通用: 禁止逐工况常数 K#"
    severity = "high"

    def check(self, ctx: VerifyContext) -> ProbeResult:
        from ...core import dsl as _dsl
        g = ctx.graph
        if g is None:
            try:
                g = _dsl.parse_dsl(ctx.dsl, list(ctx.var_names))
                ctx.graph = g
            except Exception:
                return ProbeResult(self.name, UNVERIFIED, {"reason": "parse_failed"},
                                   self.severity)
        n_c = len(g.case_names())
        ev = {"n_case_params": n_c, "case_params": list(g.case_names())}
        if n_c > 0:
            return ProbeResult(self.name, VIOLATE, ev, self.severity,
                               min_fix=f"删除逐工况常数 {list(g.case_names())}, "
                                       f"改用全局常数 C# + 局部量 (roloc/ret) 表达")
        return ProbeResult(self.name, PASS, ev, self.severity)


# ===========================================================================
# C5 跨 Ro 泛化 (LOSO)
# ===========================================================================
class LeaveOneCaseOutRule(Constraint):
    name = "C5_loso"
    description = "leave-one-工况-out: 只做 LOSO 泛化才可信 (跨 Ro 是真正的考验)"
    severity = "high"

    def __init__(self, n_starts: int = 2, budget_s: float = 6.0,
                 max_loss: float = 0.5, enabled: bool = False):
        self.n_starts = n_starts
        self.budget_s = budget_s
        self.max_loss = max_loss
        self.enabled = enabled

    def applies(self, ctx: VerifyContext) -> bool:
        return bool(ctx.cases) and len(ctx.cases) >= 3

    def check(self, ctx: VerifyContext) -> ProbeResult:
        if not self.enabled:
            return ProbeResult(self.name, UNVERIFIED,
                               {"reason": "loso_probe_disabled",
                                "hint": "ASMDomain(loso_probe=True) 打开"},
                               self.severity,
                               note="未做 LOSO → 跨工况结论不可下")
        from ...core import dsl as _dsl
        from ...core.paramfit import fit_joint
        try:
            g = ctx.graph or _dsl.parse_dsl(ctx.dsl, list(ctx.var_names))
        except Exception:
            return ProbeResult(self.name, UNVERIFIED, {"reason": "parse_failed"},
                               self.severity)
        cases = ctx.cases
        ls = []
        for k in range(len(cases)):
            train = [c for i, c in enumerate(cases) if i != k]
            try:
                fr = fit_joint(g, train, n_starts=self.n_starts, maxiter=80,
                               time_budget_s=self.budget_s,
                               seed=int(getattr(ctx.domain, "seed", 0) or 0))
                th = fr.theta
            except Exception:
                ls.append(float("inf")); continue
            nms = per_case_nmse(_preds_with(cases[k], g, th), [cases[k]])
            ls.append(float(nms[0]))
        finite = [x for x in ls if np.isfinite(x)]
        mean = float(np.mean(finite)) if finite else float("inf")
        worst = float(np.max(finite)) if finite else float("inf")
        ev = {"loso_per_case": [None if not np.isfinite(x) else round(x, 4) for x in ls],
              "loso_mean": round(mean, 4) if np.isfinite(mean) else None,
              "loso_worst": round(worst, 4) if np.isfinite(worst) else None,
              "threshold": self.max_loss}
        if mean > self.max_loss:
            return ProbeResult(self.name, VIOLATE, ev, self.severity,
                               min_fix="留一工况后崩 → 结构依赖单一工况, 需引入"
                                       "工况相关局部量 (roloc/ret) 而非常数拟合")
        return ProbeResult(self.name, PASS, ev, self.severity)


def _preds_with(case: dict, graph, theta) -> List[np.ndarray]:
    from ...core.compiler import CompiledGraph
    cg = CompiledGraph(graph, backend="numpy")
    th = np.asarray(theta, dtype=float)
    out = []
    try:
        out.append(np.asarray(cg.eval_case(th, case["X"]), dtype=float))
    except Exception:
        out.append(np.full(len(case["y"]), np.nan))
    return out


# ===========================================================================
# C6 量纲一致性
# ===========================================================================
class DimensionlessRule(Constraint):
    name = "C6_dimensionless"
    description = "所有自变量必须是无量纲已知量 (不变量 / Re_t / Ro_loc / yw)"
    severity = "low"

    def check(self, ctx: VerifyContext) -> ProbeResult:
        from .invariants import DEFAULT_VARS
        allowed = set(DEFAULT_VARS)
        used = set(ctx.var_names)
        unknown = sorted(used - allowed)
        ev = {"vars": sorted(used), "known": sorted(allowed), "unknown": unknown}
        if unknown:
            return ProbeResult(self.name, WARN, ev, self.severity,
                               min_fix=f"自变量 {unknown} 不在无量纲白名单内")
        return ProbeResult(self.name, PASS, ev, self.severity)


# ===========================================================================
# C7 数值鲁棒性
# ===========================================================================
class NumericRobustnessRule(Constraint):
    name = "C7_numeric"
    description = "极端输入 (yw→0 / →1, 量级外推) 下不得产生 NaN/inf/负 log 定义域错误"
    severity = "high"

    def applies(self, ctx: VerifyContext) -> bool:
        return ctx.cases is not None and bool(ctx.cases)

    def check(self, ctx: VerifyContext) -> ProbeResult:
        from ...core import dsl as _dsl
        from ...core.compiler import CompiledGraph
        g = ctx.graph
        if g is None:
            try:
                g = _dsl.parse_dsl(ctx.dsl, list(ctx.var_names)); ctx.graph = g
            except Exception:
                return ProbeResult(self.name, UNVERIFIED, {"reason": "parse_failed"},
                                   self.severity)
        try:
            cg = CompiledGraph(g, backend="numpy")
        except Exception as e:                                  # noqa: BLE001
            return ProbeResult(self.name, VIOLATE, {"compile_error": repr(e)},
                               self.severity, min_fix="图无法编译 (算子定义域问题)")
        th = np.asarray(ctx.theta if ctx.theta is not None else [], dtype=float)
        base = ctx.cases[0]["X"]
        bad = 0; tested = 0
        for scale in (0.5, 2.0, 10.0):
            for shift in (-1.0, 0.0, 1.0):
                X = np.clip(base, 1e-12, None) * scale
                X = X + shift * np.abs(base) * 0.5
                X = np.clip(X, 1e-12, None)
                tested += 1
                try:
                    yp = cg.eval_case(th, X)
                    if not np.all(np.isfinite(yp)):
                        bad += 1
                except Exception:
                    bad += 1
        ev = {"n_perturb_tested": tested, "n_failed": bad}
        if bad:
            return ProbeResult(self.name, VIOLATE, ev, self.severity,
                               min_fix="存在定义域外求值失败 → 用 log(x+eps) / "
                                       "tanh / 比值抑制奇点")
        return ProbeResult(self.name, PASS, ev, self.severity)


# ===========================================================================
# C8 复杂度
# ===========================================================================
class ComplexityRule(Constraint):
    name = "C8_complexity"
    description = "节点/参数上限 (可解释性与过拟合护栏)"
    severity = "low"

    def __init__(self, max_nodes: int = 22, max_params: int = 8):
        self.max_nodes = max_nodes
        self.max_params = max_params

    def check(self, ctx: VerifyContext) -> ProbeResult:
        from ...core import dsl as _dsl
        g = ctx.graph
        if g is None:
            try:
                g = _dsl.parse_dsl(ctx.dsl, list(ctx.var_names)); ctx.graph = g
            except Exception:
                return ProbeResult(self.name, UNVERIFIED, {"reason": "parse_failed"},
                                   self.severity)
        c = g.complexity()
        ev = {"nodes": c.get("nodes"), "params": c.get("params"),
              "limits": {"nodes": self.max_nodes, "params": self.max_params}}
        if (c.get("nodes", 0) > self.max_nodes
                or c.get("params", 0) > self.max_params):
            return ProbeResult(self.name, WARN, ev, self.severity,
                               min_fix="过于复杂, 建议削减节点/参数")
        return ProbeResult(self.name, PASS, ev, self.severity)


# ===========================================================================
# C9 反作弊
# ===========================================================================
class SynthCheatRule(Constraint):
    name = "C9_synth_cheat"
    description = ("在诱饵目标 (其它分量) 上拟合异常好 → 候选捕捉的是通用结构, "
                   "不是 b23 的物理 (对应验收 5 硬规则的'合成作弊')")
    severity = "high"

    def __init__(self, factor: float = 0.5, min_abs_loss: float = 0.3):
        self.factor = factor
        self.min_abs_loss = min_abs_loss

    def applies(self, ctx: VerifyContext) -> bool:
        ex = _extras(ctx)
        return bool(ex.get("decoy_cases"))

    def check(self, ctx: VerifyContext) -> ProbeResult:
        from ...core.compiler import CompiledGraph
        from ...core.paramfit import fit_joint
        ex = _extras(ctx)
        decoys = ex.get("decoy_cases") or {}
        if not decoys:
            return ProbeResult(self.name, UNVERIFIED, {"reason": "no_decoy"},
                               self.severity)
        from ...core import dsl as _dsl
        try:
            g = ctx.graph or _dsl.parse_dsl(ctx.dsl, list(ctx.var_names))
        except Exception:
            return ProbeResult(self.name, UNVERIFIED, {"reason": "parse_failed"},
                               self.severity)
        # 真目标的 loss (复用已给的 theta)
        main_loss = None
        try:
            cg = CompiledGraph(g, backend="numpy")
            th = np.asarray(ctx.theta, dtype=float)
            nms = per_case_nmse([np.asarray(cg.eval_case(th, c["X"]), dtype=float)
                                 for c in ctx.cases], ctx.cases)
            main_loss = float(np.mean(nms))
        except Exception:
            pass
        ev: Dict[str, Any] = {"main_loss": main_loss, "decoys": {},
                              "criterion": f"decoy_loss < {self.min_abs_loss} (确实拟合上了) "
                                           f"且 decoy_loss < {self.factor}×main_loss"}
        cheated = None
        for name, dcases in decoys.items():
            try:
                fr = fit_joint(g, dcases, n_starts=1, maxiter=60,
                               time_budget_s=4.0, seed=0)
                dl = float(fr.loss)
                ev["decoys"][name] = round(dl, 5)
                better = (main_loss is not None and np.isfinite(dl)
                          and dl < self.min_abs_loss
                          and dl < self.factor * max(main_loss, 1e-12))
                if better:
                    cheated = name
            except Exception as e:                              # noqa: BLE001
                ev["decoys"][name] = f"error:{e!r}"
        if cheated:
            return ProbeResult(self.name, VIOLATE, ev, self.severity,
                               min_fix=f"候选在诱饵目标 {cheated!r} 上拟合更好 → "
                                       f"它表达的是通用形状而非 b23 物理, 需重设结构")
        return ProbeResult(self.name, PASS, ev, self.severity)


# ===========================================================================
def asm_constraints(cfg=None, *, loso_probe: bool = False) -> List[Constraint]:
    """ASM 域 9 条约束 (C1..C9)。"""
    return [
        ParityRule(),
        TensorRepresentabilityRule(),
        RangeRule(),
        GlobalParamsOnlyRule(),
        LeaveOneCaseOutRule(enabled=bool(loso_probe)),
        DimensionlessRule(),
        NumericRobustnessRule(),
        ComplexityRule(max_nodes=int(getattr(cfg, "max_edges", 14)) + 8,
                       max_params=int(getattr(cfg, "max_params", 6)) + 2),
        SynthCheatRule(),
    ]
