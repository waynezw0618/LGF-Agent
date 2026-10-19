"""lgf_agent.agent.roles — L4 角色层: Creator (Agent A) 与 Inspector (Agent B).

文献依据 (Tacke et al. 2026, "two agents are better than one"):
    加入 Inspector 后, "真正满足全部物理约束的导出模型" 占比
    **91%→100% (Opus 4.7) / 37%→56% (Kimi K2.5)**, 且精度基本不降。

**为什么"两个 agent 优于一个"在这里成立**: 价值不在"多一个 LLM", 而在
**生成与审查的激励分离** —— 生成者的目标是拟合, 审查者的目标是合规, 两者用
不同的 system prompt、可选的**不同底座模型**, 避免自证偏差。我们的历史证据
(MIX 轮"存在 LLM 位置偏好未排除") 说明这个风险是真实的。

关键分工 (方案 §3.4):
- **Creator**: 输出"动作序列 + 候选", 不做最终裁决;
- **Inspector**: **不产生新公式**, 只裁决。裁决以**数值探针**为准, LLM 只负责
  给出 `min_fix` 与 narrative; 无法数值化的物理直觉一律标 `unverified_assumptions`。
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..projector import render as render_view
from ..store import Candidate
from ..verify import VerifyContext, Verifier
from .runtime import clip, estimate_tokens, parse_agent_response

CREATOR_SYSTEM = """你是 LGF-Agent 的 Creator (生成者). 目标: 发现能**跨工况泛化**且
**物理合规**的闭式方程, 而不是把单点拟合误差压到最低.

工作方式: 你可以**调用科学工具**来获取证据, 再据此提出候选。禁止凭空猜测数值。
输出必须是一个 JSON 对象:
{
  "thought": "简短的推理 (说明你从证据里看到什么, 打算怎么做)",
  "actions": [{"tool": "<工具名>", "args": {...}}, ...],
  "candidates": [{"dsl": "<LGF-DSL>", "rationale": "为什么这个结构"}]
}
规则:
- 一次最多 4 个 action; 工具的 observation 会在下一轮返回给你。
- candidates 只放你**真正想入池**的候选 (建议 1-4 条); 会由数值探针裁决。
- 不要输出工具返回值的复述, 只写你的推理与新候选。
"""

INSPECTOR_SYSTEM = """你是 LGF-Agent 的 Inspector (审查者). 你**不产生新公式**, 只裁决
给定候选是否满足物理与数值约束.

裁决依据是**数值探针的证据** (会以 JSON 给你), 不是你的直觉。要求:
- 证据说 violate 的, 你不得改判 pass;
- 无法数值检验的物理直觉, 放进 "unverified_assumptions", **不得**当作约束使用;
- 对每条违规给出**最小修复方向** (改哪一项、改成什么形态), 而不是泛泛建议。
输出 JSON:
{"verdict": "pass|return_for_refinement|reject",
 "violations": [{"constraint": str, "evidence": {...}, "min_fix": str}],
 "unverified_assumptions": [str],
 "comment": "一句话总结"}
"""


# ===========================================================================
class Creator:
    def __init__(self, ctx, registry, llm, *, audit=None, ledger=None,
                 domain=None, cfg=None):
        self.ctx = ctx
        self.reg = registry
        self.llm = llm
        self.audit = audit
        self.ledger = ledger
        self.domain = domain
        self.cfg = cfg or ctx.cfg

    # ---------------- prompt ----------------
    def build_prompt(self, *, pool: Sequence[Candidate], step: int, branch: int,
                     branch_prior: str = "", obs_log: Optional[List[str]] = None,
                     skills: Optional[Dict[str, str]] = None) -> str:
        view = render_view(pool, view=getattr(self.cfg, "agent_projector", "pareto"),
                           max_n=getattr(self.cfg, "agent_max_ctx_candidates", 8))
        task = ""
        if self.domain is not None and hasattr(self.domain, "task_card"):
            task = self.domain.task_card()
        skill_hint = ""
        if skills:
            skill_hint = ("\n[可用知识卡 (用 read_skill 读取)]\n  "
                          + ", ".join(sorted(skills)) + "\n")
        prior_blk = f"\n[本分支的假设族先验 —— 这是**假设**, 不是事实]\n{branch_prior}\n" \
            if branch_prior else ""
        obs_blk = ""
        if obs_log:
            obs_blk = "\n[上一步工具返回的 observation]\n" + "\n".join(obs_log[-6:])
        return (f"[STEP {step} · BRANCH {branch}]\n{task}\n"
                f"\n[可用工具]\n{self.reg.describe()}\n{skill_hint}"
                f"\n[当前候选池]\n{view}\n{prior_blk}{obs_blk}\n"
                f"\n请输出 JSON (thought / actions / candidates)。")

    # ---------------- 采样 ----------------
    def _complete(self, prompt: str, n: int = 1) -> List[str]:
        t0 = time.time()
        try:
            out = self.llm.complete(prompt, n=n)
        except Exception as e:                                  # noqa: BLE001
            if self.audit:
                self.audit.log(event="llm_error", role="creator", error=repr(e))
            return []
        if isinstance(out, str):
            out = [out]
        # 记账优先级: 服务端 usage (last_usage, estimated=False) > 字符估算。
        # 兼容三种客户端形态: usage() 方法 (BaseLLM) / last_usage 属性 / 两者皆无。
        u = None
        _get = getattr(self.llm, "usage", None)
        if callable(_get):
            u = _get()
        if not isinstance(u, dict) or not u.get("total_tokens"):
            u = getattr(self.llm, "last_usage", None)
        if isinstance(u, dict) and u.get("total_tokens"):
            toks = int(u["total_tokens"])
        else:
            toks = sum(estimate_tokens(p) for p in [prompt] + list(out))
        dt = time.time() - t0
        if self.ledger:
            self.ledger.add("creator", tokens=toks, seconds=dt)
        if self.ctx is not None:
            self.ctx.charge(tokens=float(toks), seconds=dt)
        return list(out)

    # ---------------- 主入口 ----------------
    def propose(self, *, pool: Sequence[Candidate], step: int, branch: int,
                branch_prior: str = "", k: int = 1,
                skills: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """返回 {candidates:[Candidate], observations:[str], thought, actions}。"""
        is_heuristic = bool(getattr(self.llm, "is_heuristic", lambda: False)())
        if is_heuristic:
            return self._propose_offline(pool=pool, step=step, branch=branch,
                                         branch_prior=branch_prior, skills=skills)
        admitted: List[Candidate] = []
        observations: List[str] = []
        thought, seen_actions = "", []
        for _ in range(max(1, int(k))):
            prompt = self.build_prompt(pool=pool, step=step, branch=branch,
                                       branch_prior=branch_prior,
                                       obs_log=observations, skills=skills)
            reps = self._complete(prompt, n=1)
            if not reps:
                continue
            parsed = parse_agent_response(reps[0], tool_names=self.reg.names())
            thought = parsed.get("thought") or thought
            if self.audit:
                self.audit.log(event="creator_response", role="creator",
                               step=step, branch=branch, parsed=parsed)
            # 1) 执行动作 (至多 4 个)
            for a in parsed["actions"][:4]:
                res = self.reg.dispatch(a["tool"], **a.get("args", {}))
                seen_actions.append({"tool": a["tool"], "ok": res.ok})
                observations.append(f"{a['tool']} -> {clip(res.observation, 240)}")
                if self.audit:
                    self.audit.log(event="tool_call", role="creator",
                                   step=step, tool=a["tool"], ok=res.ok,
                                   cost=res.cost)
            # 2) 物化候选
            for c in parsed["candidates"]:
                cand = self.materialize(c.get("dsl", ""), rationale=c.get("rationale", ""),
                                        origin=f"creator/step{step}/b{branch}")
                if cand is not None:
                    admitted.append(cand)
        return {"candidates": admitted, "observations": observations,
                "thought": thought, "actions": seen_actions}

    # ---------------- 离线 (heuristic) 路径 ----------------
    def _propose_offline(self, *, pool, step, branch, branch_prior, skills) -> Dict[str, Any]:
        """无 LLM 时用**域提供的种子骨架** + core.llm_ops.G_INIT 启发式模板产出候选。

        域种子优先 (让离线通道也能给出有意义的结果, 便于 CI/冒烟与 legacy 对照);
        再把 G_INIT 的通用模板作为多样性补充。
        """
        from ..core import llm_ops
        from ..core.priors import data_profile
        seeds: List[str] = []
        if self.domain is not None and hasattr(self.domain, "seed_skeletons"):
            try:
                seeds = list(self.domain.seed_skeletons(branch) or [])
            except Exception:                                   # noqa: BLE001
                seeds = []
        prof = data_profile(self.ctx.cases, self.ctx.var_names)
        tmpl = []
        try:
            from ..core.templates import list_templates
            tmpl = list_templates()
        except Exception:                                       # noqa: BLE001
            pass
        extra = f"假设族先验: {branch_prior}" if branch_prior else ""
        try:
            generic = llm_ops.G_INIT(self.llm, prof, tmpl, n=4, use_fewshot=True,
                                     extra_context=extra)
        except Exception:                                       # noqa: BLE001
            generic = []
        admitted = []
        for d in list(seeds) + list(generic):
            c = self.materialize(d, rationale="offline seed" if d in seeds
                                 else "heuristic G_INIT",
                                 origin=f"offline/step{step}/b{branch}")
            if c is not None:
                admitted.append(c)
        return {"candidates": admitted, "observations": [], "thought": "offline",
                "actions": []}

    # ---------------- 候选物化 (拟合 + 数值探针) ----------------
    def materialize(self, dsl: str, *, rationale: str = "", origin: str = ""
                    ) -> Optional[Candidate]:
        if not dsl or not str(dsl).strip():
            return None
        cfg = self.cfg
        r = self.reg.dispatch("fit_constants", dsl=dsl,
                              budget_s=cfg.gp_fit_budget_s, starts=cfg.gp_starts)
        if not r.ok:
            if self.audit:
                self.audit.log(event="candidate_rejected", reason="parse_or_fit",
                               dsl=dsl, detail=r.observation)
            return None
        theta = r.payload.get("theta")
        g = r.payload.get("graph")
        loss = float(r.payload.get("loss", float("inf")))
        gr = self.reg.dispatch("guardrail_check", dsl=dsl,
                               theta=[float(v) for v in theta])
        report = gr.payload.get("report", {}) if gr.ok else {}
        verdict = gr.payload.get("verdict", "unverified") if gr.ok else "unverified"
        cand = Candidate(
            dsl=str(dsl), var_names=list(self.ctx.var_names),
            theta=[float(v) for v in theta],
            loss=loss,
            complexity=dict(g.complexity()) if g is not None else {},
            metrics={"loss": loss},
            guardrails={k: v for k, v in report.items()},
            provenance={"op": "materialize", "origin": origin,
                        "rationale": rationale,
                        "prompt_hash": _hash(origin),
                        "tokens": int(self.ctx.ledger.get("tokens", 0))},
            status="admitted" if verdict in ("pass", "return_for_refinement")
            else "rejected",
        )
        if self.ctx.archive is not None:
            cand = self.ctx.archive.add(cand)
        if self.audit:
            self.audit.log(event="candidate", role="creator", candidate_id=cand.id,
                           dsl=cand.dsl, loss=cand.loss, verdict=verdict,
                           violations=cand.violations())
        # 记录一次"入池比例" (对标文献 91→100% 口径)
        if self.ctx.archive is not None:
            self.ctx.archive.record("admit", candidate_id=cand.id, verdict=verdict,
                                    admissible=cand.is_admissible())
        return cand


# ===========================================================================
class Inspector:
    """裁决者。**数值探针是权威**, LLM 只补 min_fix 与 assumptions。"""

    def __init__(self, ctx, verifier: Verifier, llm=None, *, audit=None,
                 ledger=None, cfg=None):
        self.ctx = ctx
        self.verifier = verifier
        self.llm = llm
        self.audit = audit
        self.ledger = ledger
        self.cfg = cfg or ctx.cfg

    def review(self, cand: Candidate) -> Dict[str, Any]:
        report = cand.guardrails or {}
        if not report:
            vctx = VerifyContext(dsl=cand.dsl, var_names=list(self.ctx.var_names),
                                 theta=cand.theta, cases=self.ctx.cases,
                                 domain=self.ctx.domain,
                                 extras=dict(self.ctx.extras or {}))
            report = self.verifier.run_all(vctx)
            cand.guardrails = report
        s = Verifier.summarize(report)
        numeric_verdict = Verifier.verdict(report)
        out = {
            "verdict": numeric_verdict,
            "numeric": s,
            "violations": [
                {"constraint": k, "evidence": report[k].get("evidence", {}),
                 "min_fix": report[k].get("min_fix", "")}
                for k in s["violations"]],
            "unverified_assumptions": list(s["unverified"]),
            "comment": "",
            "source": "numeric",
        }
        # LLM 只负责"最小修复方向"的补充判断 (不覆盖数值裁决)
        if self.llm is not None and not getattr(self.llm, "is_heuristic",
                                                lambda: False)():
            narrative = self._llm_comment(cand, report, numeric_verdict)
            if narrative:
                out["comment"] = narrative.get("comment", "")
                fix_map = {v.get("constraint"): v.get("min_fix")
                           for v in (narrative.get("violations") or [])
                           if isinstance(v, dict)}
                for v in out["violations"]:
                    if not v["min_fix"] and fix_map.get(v["constraint"]):
                        v["min_fix"] = fix_map[v["constraint"]]
                extra = narrative.get("unverified_assumptions") or []
                out["unverified_assumptions"] = sorted(
                    set(out["unverified_assumptions"]) | set(map(str, extra)))
                out["source"] = "numeric+llm"
        if self.audit:
            self.audit.log(event="inspector", role="inspector",
                           candidate_id=cand.id, verdict=out["verdict"],
                           source=out["source"], violations=s["violations"])
        if self.ctx.archive is not None:
            self.ctx.archive.record("review", candidate_id=cand.id,
                                    verdict=out["verdict"])
        return out

    def _llm_comment(self, cand: Candidate, report: Dict[str, Any],
                     numeric_verdict: str) -> Optional[Dict[str, Any]]:
        import json as _json
        prompt = (f"[INSPECTOR] 候选: {cand.dsl}\n"
                  f"数值探针报告 (权威):\n{_json.dumps(report, ensure_ascii=False, default=str)[:2400]}\n"
                  f"数值裁决: {numeric_verdict}\n"
                  f"请按 system 要求输出 JSON。")
        t0 = time.time()
        try:
            reps = self.llm.complete(prompt, n=1)
        except Exception as e:                                  # noqa: BLE001
            if self.audit:
                self.audit.log(event="llm_error", role="inspector", error=repr(e))
            return None
        text = reps[0] if isinstance(reps, list) and reps else (reps if isinstance(reps, str) else "")
        if not text:
            return None
        u = None
        _get = getattr(self.llm, "usage", None)
        if callable(_get):
            u = _get()
        if not isinstance(u, dict) or not u.get("total_tokens"):
            u = getattr(self.llm, "last_usage", None)
        if isinstance(u, dict) and u.get("total_tokens"):
            toks = int(u["total_tokens"])
        else:
            toks = estimate_tokens(prompt) + estimate_tokens(text)
        dt = time.time() - t0
        if self.ledger:
            self.ledger.add("inspector", tokens=toks, seconds=dt)
        self.ctx.charge(tokens=float(toks), seconds=dt)
        return parse_agent_response(text, tool_names=())


def _hash(s: str) -> str:
    import hashlib
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()[:12]
