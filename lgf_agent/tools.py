"""lgf_agent.tools — L1 动作层: 统一科学动作接口.

诊断 (方案 §1.1)
---------------
`core/llm_ops.py` 注册了 5 个外层算子, 实际只有 2 个被接线 (G_INIT / M_GUIDE),
`T_PROP` / `H_DECOMP` / `E_CHECK` 是死代码。而 LGF **已经具备**这些能力
(`paramfit.fit_joint` 拟合、`compiler.CompiledGraph` 求值、`priors.run_priors`
先验、`eq_compat` 符号等价、`gp.py` 爬山), 只是**没暴露给 LLM**。

本层的结论: 升级的 80% 工作是"接线", 不是造新轮子。

统一接口
--------
    ToolResult(ok, observation, evidence, cost)
      observation : 进上下文的紧凑文本 (≤400 字符) —— 对应 SRHarness
                    "runtime 保留详细证据但不塞进上下文"
      evidence    : 全量机器可读证据 (落 archive)
      cost        : {seconds, tokens, calls} (成本会计)

关键: **表达式视图** (SRHarness 消融掉幅最大的一档, 74.47%→67.57%)。
我们有天然优势: `attribution.analyze` 本来就在算每个节点的子树值 `z_v`,
把 `z_v` 提升为一等公民 (`residual_view` 工具) 几乎零成本。
"""
from __future__ import annotations

import inspect
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .store import Candidate, Archive

OBS_LIMIT = 400


def _clip(s: str, n: int = OBS_LIMIT) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


@dataclass
class ToolResult:
    ok: bool
    observation: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    cost: Dict[str, Any] = field(default_factory=dict)
    # 供调度器内部使用 (不注入 prompt)
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dict(ok=self.ok, observation=self.observation,
                    evidence=self.evidence, cost=self.cost)


# ---------------------------------------------------------------------------
@dataclass
class ToolContext:
    """一次 agent 运行内所有工具共享的上下文."""
    cases: List[dict]
    var_names: List[str]
    cfg: Any
    archive: Optional[Archive] = None
    domain: Any = None
    verifier: Any = None
    llm_client: Any = None
    rng: Any = None
    skills: Dict[str, str] = field(default_factory=dict)
    # 域提供的附加证据 (case_extras / decoy_cases), 传给 VerifyContext.extras
    extras: Dict[str, Any] = field(default_factory=dict)
    out_dir: str = "outputs"
    ledger: Dict[str, float] = field(default_factory=lambda: {"tokens": 0.0,
                                                              "usd": 0.0,
                                                              "calls": 0.0,
                                                              "seconds": 0.0})

    def charge(self, **kv: float) -> None:
        for k, v in kv.items():
            if k in self.ledger:
                self.ledger[k] += float(v)

    def budget_exhausted(self) -> Optional[str]:
        if self.ledger["tokens"] > getattr(self.cfg, "agent_token_budget", 10 ** 9):
            return "token_budget"
        if self.ledger["usd"] > getattr(self.cfg, "agent_cost_budget_usd", 1e9):
            return "cost_budget"
        return None


class Tool:
    """工具基类。`schema()` 的返回文本会注入 prompt (供 LLM 选择动作)."""
    name = "tool"
    description = ""
    # 入参说明: {arg: (type_str, required, 说明)}
    args: Dict[str, tuple] = {}

    def __init__(self, ctx: ToolContext):
        self.ctx = ctx

    def schema(self) -> str:
        parts = []
        for a, (t, req, desc) in self.args.items():
            parts.append(f"{a}:{t}{'' if req else '?'} — {desc}")
        return f"- {self.name}({'; '.join(parts)}) :: {self.description}"

    def __call__(self, **kwargs: Any) -> ToolResult:   # pragma: no cover
        raise NotImplementedError


# ===========================================================================
# 1. profile_data
# ===========================================================================
class ProfileData(Tool):
    name = "profile_data"
    description = "数据画像: 变量值域/量级/单调性 (Spearman) + 目标统计"
    args = {}

    def __call__(self, **kw) -> ToolResult:
        from .core.priors import data_profile, profile_summary
        t0 = time.time()
        prof = data_profile(self.ctx.cases, self.ctx.var_names)
        txt = profile_summary(prof)
        return ToolResult(True, _clip(txt), {"profile": prof},
                          {"seconds": time.time() - t0}, {"profile": prof})


# ===========================================================================
# 2. prior_check
# ===========================================================================
class PriorCheck(Tool):
    name = "prior_check"
    description = ("AI-Feynman 风格结构检验: 对称 / 可加可分 / 可乘可分 / 组合性, "
                   "返回 GP 可用约束")
    args = {"max_points": ("int", False, "代理拟合点数上限 (默认 600)")}

    def __call__(self, max_points: int = 600, **kw) -> ToolResult:
        from .core.priors import run_priors
        t0 = time.time()
        cases = self.ctx.cases
        # 子采样: 代理场拟合需要稠密点, 但全量可能过慢
        sub = []
        rng = self.ctx.rng or np.random.default_rng(0)
        tot = sum(len(c["X"]) for c in cases)
        frac = min(1.0, float(max_points) / max(1, tot))
        for c in cases:
            n = len(c["X"])
            idx = np.arange(n)
            if frac < 1.0:
                idx = np.sort(rng.choice(n, max(8, int(n * frac)), replace=False))
            d = dict(c); d["X"] = c["X"][idx]; d["y"] = c["y"][idx]
            sub.append(d)
        try:
            report, pc = run_priors(sub, self.ctx.var_names)
        except Exception as e:                              # noqa: BLE001
            return ToolResult(False, f"先验检验失败: {e!r}",
                              {"error": repr(e)},
                              {"seconds": time.time() - t0})
        sym = [f"{self.ctx.var_names[i]}/{self.ctx.var_names[j]}"
               for (i, j), (ok, _) in (report.get("symmetry") or {}).items() if ok]
        obs = (f"对称对: {sym or '无'}; 可加可分: "
               f"{[[f'{self.ctx.var_names[i]}',f'{self.ctx.var_names[j]}'] for i, j in pc.additive_partitions] or '无'}; "
               f"可乘可分: "
               f"{[[f'{self.ctx.var_names[i]}',f'{self.ctx.var_names[j]}'] for i, j in pc.multiplicative_partitions] or '无'}")
        return ToolResult(True, _clip(obs),
                          {"report": _jsonable(report), "constraints": pc.to_dict()},
                          {"seconds": time.time() - t0},
                          {"prior_constraints": pc})


# ===========================================================================
# 3. fit_constants
# ===========================================================================
class FitConstants(Tool):
    name = "fit_constants"
    description = "对给定 DSL 做多启动 L-BFGS-B 联合拟合, 返回 loss/theta/收敛性"
    args = {
        "dsl": ("str", True, "LGF-DSL 表达式"),
        "budget_s": ("float", False, "拟合时间预算 (默认 cfg.gp_fit_budget_s)"),
        "starts": ("int", False, "多启动次数 (默认 cfg.gp_starts)"),
    }

    def __call__(self, dsl: str, budget_s: Optional[float] = None,
                 starts: Optional[int] = None, **kw) -> ToolResult:
        from .core import dsl as _dsl
        from .core.paramfit import fit_joint
        cfg = self.ctx.cfg
        t0 = time.time()
        try:
            g = _dsl.parse_dsl(dsl, self.ctx.var_names)
        except Exception as e:                              # noqa: BLE001
            return ToolResult(False, f"DSL 解析失败: {e!r}", {"error": repr(e)},
                              {"seconds": time.time() - t0})
        res = fit_joint(g, self.ctx.cases,
                        n_starts=int(starts or cfg.gp_starts),
                        maxiter=int(cfg.final_maxiter),
                        time_budget_s=float(budget_s or cfg.gp_fit_budget_s),
                        seed=int(getattr(cfg, "seed", 0)))
        obs = (f"loss={res.loss:.5g} theta={np.round(res.theta,4).tolist()} "
               f"finite={res.n_finite} ok={res.success}")
        return ToolResult(True, _clip(obs),
                          {"loss": res.loss, "theta": [float(v) for v in res.theta],
                           "per_case": res.per_case, "n_finite": res.n_finite,
                           "success": bool(res.success)},
                          {"seconds": time.time() - t0},
                          {"graph": g, "theta": res.theta, "loss": res.loss})


# ===========================================================================
# 4. evaluate_dsl
# ===========================================================================
class EvaluateDsl(Tool):
    name = "evaluate_dsl"
    description = "逐工况 NMSE / R² / 值域 (给定 dsl 与 theta)"
    args = {"dsl": ("str", True, "LGF-DSL"),
            "theta": ("list", False, "参数向量 (省略则先拟合)")}

    def __call__(self, dsl: str, theta: Optional[Sequence[float]] = None,
                 **kw) -> ToolResult:
        from .core import dsl as _dsl
        from .core.paramfit import fit_joint, evaluate
        t0 = time.time()
        try:
            g = _dsl.parse_dsl(dsl, self.ctx.var_names)
        except Exception as e:                              # noqa: BLE001
            return ToolResult(False, f"DSL 解析失败: {e!r}", {"error": repr(e)},
                              {"seconds": time.time() - t0})
        th = np.asarray(theta if theta is not None else [], dtype=float)
        if th.size == 0:
            fr = fit_joint(g, self.ctx.cases, n_starts=self.ctx.cfg.gp_starts,
                           maxiter=self.ctx.cfg.final_maxiter,
                           time_budget_s=self.ctx.cfg.gp_fit_budget_s,
                           seed=int(getattr(self.ctx.cfg, "seed", 0)))
            th = fr.theta
        m = evaluate(g, th, self.ctx.cases)
        nms = {k: v for k, v in m.items() if k.endswith("_nmse")}
        obs = " ".join(f"{k.replace('_nmse','')}:{v:.3g}" for k, v in nms.items())
        return ToolResult(True, _clip("逐工况 NMSE " + obs),
                          {"metrics": m, "theta": [float(v) for v in th]},
                          {"seconds": time.time() - t0},
                          {"metrics": m, "theta": th})


# ===========================================================================
# 5. attribute
# ===========================================================================
class Attribute(Tool):
    name = "attribute"
    description = "节点级归因: 各节点敏感度 top-5 + 形状标签 (驱动定向变异)"
    args = {"dsl": ("str", True, "LGF-DSL"),
            "theta": ("list", False, "参数向量 (省略则先拟合)")}

    def __call__(self, dsl: str, theta: Optional[Sequence[float]] = None,
                 **kw) -> ToolResult:
        from .core import dsl as _dsl
        from .core import attribution
        from .core.paramfit import fit_joint
        t0 = time.time()
        try:
            g = _dsl.parse_dsl(dsl, self.ctx.var_names)
        except Exception as e:                              # noqa: BLE001
            return ToolResult(False, f"DSL 解析失败: {e!r}", {"error": repr(e)},
                              {"seconds": time.time() - t0})
        th = np.asarray(theta if theta is not None else [], dtype=float)
        if th.size == 0:
            fr = fit_joint(g, self.ctx.cases, n_starts=self.ctx.cfg.gp_starts,
                           maxiter=self.ctx.cfg.final_maxiter,
                           time_budget_s=self.ctx.cfg.gp_fit_budget_s,
                           seed=int(getattr(self.ctx.cfg, "seed", 0)))
            th = fr.theta
        try:
            rep = attribution.analyze(g, th, self.ctx.cases, self.ctx.var_names)
        except Exception as e:                              # noqa: BLE001
            return ToolResult(False, f"归因失败: {e!r}", {"error": repr(e)},
                              {"seconds": time.time() - t0})
        txt = rep.get("text_summary") or rep.get("summary") or ""
        return ToolResult(True, _clip(txt or json.dumps(rep, default=str)),
                          {"attribution": _jsonable(rep)},
                          {"seconds": time.time() - t0}, {"attribution": rep})


# ===========================================================================
# 6. residual_view  —— 表达式视图 (SRHarness 掉幅最大的一档)
# ===========================================================================
class ResidualView(Tool):
    name = "residual_view"
    description = ("表达式视图: 残差 y-f(x) 的统计与分箱误差; 残差可作为下一轮 "
                   "fit_constants 的 y (结构化残差分析)")
    args = {"dsl": ("str", True, "LGF-DSL"),
            "theta": ("list", False, "参数向量 (省略则先拟合)"),
            "bins": ("int", False, "分箱数 (默认 8)")}

    def __call__(self, dsl: str, theta: Optional[Sequence[float]] = None,
                 bins: int = 8, **kw) -> ToolResult:
        from .core import dsl as _dsl
        from .core.compiler import CompiledGraph
        from .core.paramfit import fit_joint
        t0 = time.time()
        try:
            g = _dsl.parse_dsl(dsl, self.ctx.var_names)
        except Exception as e:                              # noqa: BLE001
            return ToolResult(False, f"DSL 解析失败: {e!r}", {"error": repr(e)},
                              {"seconds": time.time() - t0})
        th = np.asarray(theta if theta is not None else [], dtype=float)
        if th.size == 0:
            fr = fit_joint(g, self.ctx.cases, n_starts=self.ctx.cfg.gp_starts,
                           maxiter=self.ctx.cfg.final_maxiter,
                           time_budget_s=self.ctx.cfg.gp_fit_budget_s,
                           seed=int(getattr(self.ctx.cfg, "seed", 0)))
            th = fr.theta
        cg = CompiledGraph(g, backend="numpy")
        n_g, n_c = cg.n_g, cg.n_c
        per_case = {}
        resid_cases: List[dict] = []
        for ci, case in enumerate(self.ctx.cases):
            tl = np.concatenate([th[:n_g], th[n_g + ci * n_c: n_g + (ci + 1) * n_c]])
            try:
                yp = np.asarray(cg.eval_case(tl, case["X"]), dtype=float)
            except Exception:
                continue
            y = np.asarray(case["y"], dtype=float)
            r = y - yp
            name = case.get("name", f"c{ci}")
            # 分箱误差: 按主自变量 (第一列) 分位分箱
            x0 = case["X"][:, 0]
            qs = np.quantile(x0, np.linspace(0, 1, int(bins) + 1))
            bin_err = []
            for b in range(int(bins)):
                m = (x0 >= qs[b]) & (x0 <= qs[b + 1])
                if m.sum() > 0:
                    bin_err.append(round(float(np.sqrt(np.mean(r[m] ** 2))), 5))
            per_case[name] = dict(resid_mean=float(np.mean(r)),
                                  resid_std=float(np.std(r)),
                                  resid_rms=float(np.sqrt(np.mean(r ** 2))),
                                  bin_rms=bin_err)
            resid_cases.append(dict(X=case["X"], y=r,
                                    name=name + "_resid",
                                    **({"w": case["w"]} if "w" in case else {})))
        obs = "; ".join(f"{k}: mean={v['resid_mean']:+.3g} rms={v['resid_rms']:.3g} "
                        f"bin={v['bin_rms']}" for k, v in per_case.items())
        return ToolResult(True, _clip("残差 " + obs),
                          {"per_case": per_case, "theta": [float(v) for v in th]},
                          {"seconds": time.time() - t0},
                          {"resid_cases": resid_cases, "theta": th})


# ===========================================================================
# 7. guardrail_check
# ===========================================================================
class GuardrailCheck(Tool):
    name = "guardrail_check"
    description = "运行全部数值探针约束, 逐条返回 pass/violate/unverified + 证据"
    args = {"dsl": ("str", True, "LGF-DSL"),
            "theta": ("list", False, "参数向量 (省略则先拟合)")}

    def __call__(self, dsl: str, theta: Optional[Sequence[float]] = None,
                 **kw) -> ToolResult:
        from .verify import VerifyContext, Verifier
        from .core.paramfit import fit_joint
        from .core import dsl as _dsl
        t0 = time.time()
        th = np.asarray(theta if theta is not None else [], dtype=float)
        g = None
        if th.size == 0:
            try:
                g = _dsl.parse_dsl(dsl, self.ctx.var_names)
                fr = fit_joint(g, self.ctx.cases, n_starts=self.ctx.cfg.gp_starts,
                               maxiter=self.ctx.cfg.final_maxiter,
                               time_budget_s=self.ctx.cfg.gp_fit_budget_s,
                               seed=int(getattr(self.ctx.cfg, "seed", 0)))
                th = fr.theta
            except Exception:                               # noqa: BLE001
                pass
        ver: Verifier = self.ctx.verifier
        if ver is None:
            return ToolResult(False, "未配置 verifier", {}, {"seconds": 0.0})
        ctx = VerifyContext(dsl=dsl, var_names=self.ctx.var_names, theta=th,
                            cases=self.ctx.cases, graph=g, domain=self.ctx.domain,
                            extras=dict(self.ctx.extras or {}))
        report = ver.run_all(ctx)
        s = Verifier.summarize(report)
        bad = [f"{k}:{report[k]['verdict']}" for k in s["violations"] + s["warnings"]]
        obs = (f"通过率 {s['pass_rate']:.0%} ({s['counts']}); "
               f"违规={s['violations'] or '无'}; 待验证假设={s['unverified'] or '无'}")
        return ToolResult(True, _clip(obs), {"report": report, "summary": s},
                          {"seconds": time.time() - t0},
                          {"report": report, "summary": s, "theta": th,
                           "verdict": Verifier.verdict(report)})


# ===========================================================================
# 8. symbol_equiv
# ===========================================================================
class SymbolEquiv(Tool):
    name = "symbol_equiv"
    description = "两式符号等价判定 (SA 口径) + 与金标准比对"
    args = {"dsl_a": ("str", True, "式 A"), "dsl_b": ("str", True, "式 B")}

    def __call__(self, dsl_a: str, dsl_b: str, **kw) -> ToolResult:
        from .core import dsl as _dsl
        from .core.compiler import CompiledGraph
        t0 = time.time()
        try:
            ga = _dsl.parse_dsl(dsl_a, self.ctx.var_names)
            gb = _dsl.parse_dsl(dsl_b, self.ctx.var_names)
        except Exception as e:                              # noqa: BLE001
            return ToolResult(False, f"DSL 解析失败: {e!r}", {"error": repr(e)},
                              {"seconds": time.time() - t0})
        method, equiv, ev = "none", False, {}
        # (a) 符号判定: simplify(f_a - f_b) == 0  (常数取舍一, 故仅对无参数式有效)
        try:
            import sympy as sp
            pa = [p for p in ga.params()]
            pb = [p for p in gb.params()]
            if not pa and not pb:
                diff = sp.simplify(ga.to_sympy() - gb.to_sympy())
                equiv = bool(diff == 0)
                method = "symbolic"
                ev["simplified_diff"] = str(diff)[:120]
        except Exception:                                   # noqa: BLE001
            pass
        # (b) 数值判定: 在随机点上比较 (需两者都无自由参数或已给初值)
        if method == "none":
            try:
                cga, cgb = CompiledGraph(ga), CompiledGraph(gb)
                rng = self.ctx.rng or np.random.default_rng(0)
                if cga.n_g + cga.n_c == 0 and cgb.n_g + cgb.n_c == 0:
                    X = rng.uniform(0.5, 1.5, size=(256, len(self.ctx.var_names)))
                    ya = np.asarray(cga.eval_case([], X), dtype=float)
                    yb = np.asarray(cgb.eval_case([], X), dtype=float)
                    rel = float(np.max(np.abs(ya - yb))
                                / (np.max(np.abs(ya)) + 1e-12))
                    equiv = rel < float(getattr(self.ctx.cfg, "equivalence_tol", 1e-3))
                    method = "numeric"
                    ev["max_rel_err"] = rel
                else:
                    method = "skipped_has_free_params"
            except Exception as e:                          # noqa: BLE001
                ev["numeric_error"] = repr(e)
        return ToolResult(True, _clip(f"等价={equiv} (方法={method})"),
                          {"equivalent": equiv, "method": method, **ev},
                          {"seconds": time.time() - t0}, {"equivalent": equiv})


# ===========================================================================
# 9. gp_search —— GP 降格为一个动作
# ===========================================================================
class GpSearch(Tool):
    name = "gp_search"
    description = ("在图空间做 GP 进化 (crossover/mutate/select), 返回新 top-k 候选; "
                   "GP 是工具, 不是主循环")
    args = {"parents": ("list[str]", True, "父代 DSL 列表"),
            "generations": ("int", False, "代数 (默认 8)"),
            "population": ("int", False, "种群 (默认 cfg.population)")}

    def __call__(self, parents: Sequence[str], generations: int = 8,
                 population: Optional[int] = None, **kw) -> ToolResult:
        from .core import dsl as _dsl
        from .core import gp as gp_ops
        from .core.paramfit import fit_joint
        cfg = self.ctx.cfg
        t0 = time.time()
        rng = self.ctx.rng or np.random.default_rng(int(getattr(cfg, "seed", 0)))
        pop = []
        for s in (parents or []):
            try:
                pop.append(_dsl.parse_dsl(s, self.ctx.var_names))
            except Exception:                               # noqa: BLE001
                continue
        if not pop:
            return ToolResult(False, "无有效父代 DSL", {},
                              {"seconds": time.time() - t0})
        pop_n = int(population or cfg.population)

        def _fit(g):
            return fit_joint(g, self.ctx.cases, n_starts=cfg.gp_starts,
                             maxiter=cfg.gp_maxiter,
                             time_budget_s=cfg.gp_fit_budget_s,
                             seed=int(getattr(cfg, "seed", 0)))

        scored = [(_fit(g), g) for g in pop]
        deadline = time.time() + float(cfg.time_budget_s)
        for _ in range(int(generations)):
            if time.time() > deadline:
                break
            children = []
            n_need = max(1, pop_n - len(pop))
            for _try in range(n_need * 8):
                if len(children) >= n_need:
                    break
                a = pop[int(rng.integers(len(pop)))]
                b = pop[int(rng.integers(len(pop)))]
                ch = gp_ops.crossover(a, b, rng, max_edges=cfg.max_edges,
                                      max_params=cfg.max_params,
                                      max_depth=cfg.max_depth)
                if ch is None:                              # crossover 可能返回 None
                    continue
                try:
                    ch = gp_ops.mutate(ch, rng, self.ctx.var_names,
                                       max_edges=cfg.max_edges,
                                       max_params=cfg.max_params,
                                       max_depth=cfg.max_depth)
                except Exception:                           # noqa: BLE001
                    pass
                # 关键修复 (V4b): 子代必须重置 loss/theta, 否则 GP 不进化
                ch.loss = float("inf")
                ch.theta = None
                children.append(ch)
            scored = []
            for g in pop + children:
                if not np.isfinite(getattr(g, "loss", float("inf"))):
                    g.loss = _fit(g).loss
                scored.append((g.loss, g))
            scored.sort(key=lambda t: t[0])
            keep = max(2, int(pop_n * 0.5))
            pop = [g for _, g in scored[:keep]]
        top = []
        for sc, g in sorted(scored, key=lambda t: t[0])[:5]:
            try:
                top.append({"dsl": _dsl.graph_to_dsl(g), "loss": float(sc),
                            "complexity": g.complexity()})
            except Exception:                               # noqa: BLE001
                continue
        obs = " | ".join(f"{t['loss']:.4g}:{t['dsl'][:60]}" for t in top)
        return ToolResult(True, _clip("GP 新 top: " + obs), {"top": top},
                          {"seconds": time.time() - t0}, {"top": top})


# ===========================================================================
# 10. submit_formula
# ===========================================================================
class SubmitFormula(Tool):
    name = "submit_formula"
    description = "提交候选: 冻结 + 最终精修 (final_starts) + 入 archive"
    args = {"dsl": ("str", True, "LGF-DSL"),
            "rationale": ("str", False, "选择理由 (写入 provenance)"),
            "theta": ("list", False, "参数向量 (省略则最终精修)")}

    def __call__(self, dsl: str, rationale: str = "",
                 theta: Optional[Sequence[float]] = None, **kw) -> ToolResult:
        from .core import dsl as _dsl
        from .core.paramfit import fit_joint
        cfg = self.ctx.cfg
        t0 = time.time()
        try:
            g = _dsl.parse_dsl(dsl, self.ctx.var_names)
        except Exception as e:                              # noqa: BLE001
            return ToolResult(False, f"DSL 解析失败: {e!r}", {"error": repr(e)},
                              {"seconds": time.time() - t0})
        th = np.asarray(theta if theta is not None else [], dtype=float)
        fr = None
        if th.size == 0:
            fr = fit_joint(g, self.ctx.cases, n_starts=cfg.final_starts,
                           maxiter=cfg.final_maxiter,
                           time_budget_s=cfg.time_budget_s,
                           seed=int(getattr(cfg, "seed", 0)))
            th = fr.theta
        cand = Candidate(dsl=dsl, var_names=list(self.ctx.var_names),
                         theta=[float(v) for v in th],
                         loss=float(fr.loss if fr else float("nan")),
                         complexity=dict(g.complexity()),
                         provenance={"op": "submit_formula", "rationale": rationale},
                         status="submitted")
        if self.ctx.archive is not None:
            self.ctx.archive.add(cand)
        return ToolResult(True, _clip(f"已提交 {cand.id} loss={cand.loss:.5g}"),
                          {"candidate": cand.to_dict()},
                          {"seconds": time.time() - t0}, {"candidate": cand})


# ===========================================================================
# 11. read_skill
# ===========================================================================
class ReadSkill(Tool):
    name = "read_skill"
    description = "读取领域知识卡 (张量基代数 / Pope 不变量 / 对数律 ...)"
    args = {"topic": ("str", True, "主题名")}

    def __call__(self, topic: str, **kw) -> ToolResult:
        key = str(topic).strip()
        cards = self.ctx.skills or {}
        body = cards.get(key)
        if body is None:
            # 模糊匹配
            for k, v in cards.items():
                if key and (key in k or k in key):
                    key, body = k, v
                    break
        if body is None:
            return ToolResult(False, f"无此知识卡: {topic} (可用: {sorted(cards)})",
                              {"available": sorted(cards)}, {"seconds": 0.0})
        return ToolResult(True, _clip(body), {"topic": key, "card": body},
                          {"seconds": 0.0}, {"topic": key, "card": body})


# ===========================================================================
# 12. code_exec (默认关闭)
# ===========================================================================
class CodeExec(Tool):
    name = "code_exec"
    description = "沙箱内执行 python 片段 (默认关闭, 需 cfg.agent_code_exec=True)"
    args = {"code": ("str", True, "python 片段")}

    def __call__(self, code: str, **kw) -> ToolResult:
        if not getattr(self.ctx.cfg, "agent_code_exec", False):
            return ToolResult(False, "code_exec 已关闭 (cfg.agent_code_exec=False)",
                              {"disabled": True}, {"seconds": 0.0})
        t0 = time.time()
        import contextlib
        import io
        buf = io.StringIO()
        env = {"np": np, "cases": self.ctx.cases, "var_names": self.ctx.var_names}
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, {"__builtins__": __builtins__}, env)   # noqa: S102
        except Exception as e:                              # noqa: BLE001
            return ToolResult(False, f"执行异常: {e!r}", {"error": repr(e)},
                              {"seconds": time.time() - t0})
        return ToolResult(True, _clip(buf.getvalue() or "(无输出)"),
                          {"stdout": buf.getvalue()},
                          {"seconds": time.time() - t0})


# ===========================================================================
def default_tools(ctx: ToolContext, *, include_code_exec: bool = False
                  ) -> "ToolRegistry":
    reg = ToolRegistry(ctx)
    for T in (ProfileData, PriorCheck, FitConstants, EvaluateDsl, Attribute,
              ResidualView, GuardrailCheck, SymbolEquiv, GpSearch, SubmitFormula,
              ReadSkill):
        reg.register(T(ctx))
    if include_code_exec:
        reg.register(CodeExec(ctx))
    return reg


class ToolRegistry:
    """工具注册表 + 统一 dispatch (含成本会计与审计)."""

    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self._tools: Dict[str, Tool] = {}
        self.trace: List[Dict[str, Any]] = []

    def register(self, tool: Tool) -> "ToolRegistry":
        self._tools[tool.name] = tool
        return self

    def names(self) -> List[str]:
        return list(self._tools)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def describe(self) -> str:
        return "\n".join(t.schema() for t in self._tools.values())

    def dispatch(self, name: str, **args: Any) -> ToolResult:
        t0 = time.time()
        tool = self._tools.get(name)
        if tool is None:
            res = ToolResult(False, f"未知工具 {name!r} (可用: {self.names()})",
                             {"error": "unknown_tool"})
        else:
            try:
                res = tool(**args)
            except TypeError as e:                          # 参数不匹配
                res = ToolResult(False, f"参数错误: {e}", {"error": repr(e)})
            except Exception as e:                          # noqa: BLE001
                res = ToolResult(False, f"工具异常: {e!r}", {"error": repr(e)})
        res.cost.setdefault("seconds", round(time.time() - t0, 4))
        self.ctx.charge(seconds=res.cost.get("seconds", 0.0), calls=1.0)
        self.trace.append({"tool": name, "args": _jsonable(args),
                           "ok": res.ok, "cost": res.cost,
                           "observation": res.observation})
        return res


def _jsonable(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    try:
        json.dumps(o)
        return o
    except Exception:                                       # noqa: BLE001
        return str(o)
