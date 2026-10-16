"""lgf_agent.agent.scheduler — L3 轨迹层: R-C-L-K 显式化调度器.

现状映射 (方案 §1.3): LGF 把 R-C-L-K 写死成 **1-1-N-1**
    R(重启)=无, C(分支)=无, L=generations(被 time_budget 掐断), K=1。

SRHarness 轨迹消融 (LSR-Transform-Anon, 基线 SA 74.47%):
    R1C1L30K1 74.47% / R2C1L15K1 66.67% / R1C2L15K1 72.07% / R1C1L15K2 73.87%
→ **同样预算下, 加深精炼 > 增加轨迹多样性**。这条直接否决"多跑几个种子/多开几条
  分支"的直觉方案。

调度循环 (对应 SRHarness Algorithm 1):
    1. S ← archive; 对 r ∈ 1..R: 取 TopK(S) 作为初始上下文, 新对话
    2. 对 c ∈ 1..C: 复制 prompt 骨架 → 独立分支 (分支赋不同**假设族先验**)
    3. 对 ℓ ∈ 1..L: 组 prompt (注入 P_ρ(S)) → 采样 K 个候选 → 过 Inspector 与
       动作 → Admit 进 S → 选最佳继续
    4. 早停: 提交公式通过全部约束且跨工况 worst-case 达标
    5. 全程 Record(r,c,ℓ) + provenance 落盘

预算: `time_budget_s` + `agent_token_budget` + `agent_cost_budget_usd` 三元硬上限。
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence

from ..projector import project, render
from ..store import Archive, Candidate
from ..tools import ToolContext, default_tools
from ..verify import Verifier, default_hard_rules
from .roles import Creator, Inspector
from .runtime import Audit, CostLedger


def build_verifier(domain, cfg) -> Verifier:
    """域约束 + 通用硬规则 → 一个 Verifier。"""
    cons: List[Any] = []
    if domain is not None and hasattr(domain, "constraints"):
        cons.extend(domain.constraints(cfg) or [])
    cons.extend(default_hard_rules(
        forbidden=getattr(cfg, "forbidden_ops", None) or [],
        blowup_threshold=1e3,
        cross_case_ratio=5.0))
    return Verifier(cons)


def run_agent(domain, cfg, *, llm_client=None, inspector_client=None,
              out_dir: Optional[str] = None, task_name: Optional[str] = None,
              replay_legacy: Optional[str] = None,
              resume: bool = False) -> Dict[str, Any]:
    """LGF-Agent 主入口 (并列引擎; legacy 见 core.pipeline_lgf.run_lgf)。

    Parameters
    ----------
    domain : DomainAdapter
        提供 cases / var_names / constraints / task_card / branch_priors。
    cfg : LGFConfig
        agent_R / agent_C / agent_L / agent_K / agent_projector / agent_inspector ...
    llm_client : 可省略 (省略则用 heuristic 离线通道, 管线仍可跑通)。
    replay_legacy : 若给定目录, 先把该目录下 `*_final.json` 回灌进 archive (P0 验收)。
    """
    t_start = time.time()
    if llm_client is None:
        from ..core.llm_ops import make_client
        llm_client = make_client("heuristic")

    task = task_name or getattr(domain, "name", "task")
    base_dir = os.path.join(out_dir or getattr(cfg, "output_dir", "outputs"),
                            "agent", task)
    os.makedirs(base_dir, exist_ok=True)
    archive_path = os.path.join(base_dir, "archive.jsonl")
    traj_path = os.path.join(base_dir, "traj.jsonl")

    # ---- 域数据 ----
    cases, var_names = domain.build_cases(cfg)
    archive = Archive(archive_path, task=task)
    if replay_legacy and not resume:
        legacy = Archive.from_legacy_runs(replay_legacy, task=task,
                                          archive_path=archive_path)
        for c in legacy.all():
            try:
                archive.add(c)
            except ValueError:
                archive.update(c)

    verifier = build_verifier(domain, cfg)
    audit = Audit(traj_path)
    ledger = CostLedger()

    skills = {}
    if hasattr(domain, "skills"):
        skills = domain.skills(getattr(cfg, "agent_skills", None) or None)

    ctx = ToolContext(cases=cases, var_names=list(var_names), cfg=cfg,
                      archive=archive, domain=domain, verifier=verifier,
                      llm_client=llm_client, rng=None, skills=skills,
                      extras=(domain.extras() if (domain is not None
                                                  and hasattr(domain, "extras"))
                              else {}),
                      out_dir=base_dir)
    import numpy as _np
    ctx.rng = _np.random.default_rng(int(getattr(cfg, "seed", 0)))
    registry = default_tools(ctx, include_code_exec=getattr(cfg, "agent_code_exec",
                                                            False))

    creator = Creator(ctx, registry, llm_client, audit=audit, ledger=ledger,
                      domain=domain, cfg=cfg)
    inspector = None
    if getattr(cfg, "agent_inspector", True):
        ins_llm = inspector_client if inspector_client is not None else llm_client
        inspector = Inspector(ctx, verifier, ins_llm, audit=audit, ledger=ledger,
                              cfg=cfg)

    R = max(1, int(getattr(cfg, "agent_R", 1)))
    C = max(1, int(getattr(cfg, "agent_C", 1)))
    L = max(1, int(getattr(cfg, "agent_L", 30)))
    K = max(1, int(getattr(cfg, "agent_K", 1)))
    max_ctx = int(getattr(cfg, "agent_max_ctx_candidates", 8))
    view = getattr(cfg, "agent_projector", "pareto")

    branch_priors = _branch_priors(domain, cfg, C)
    deadline = t_start + float(getattr(cfg, "time_budget_s", 600.0))

    step_log: List[Dict[str, Any]] = []
    stop_reason = "completed"
    n_admitted = n_rejected = n_reviewed = 0
    best_submitted: Optional[Candidate] = None

    for r in range(R):
        pool = project(archive.all(), view=view, max_n=max_ctx,
                       shuffle=False) if len(archive) else []
        for c in range(C):
            prior = branch_priors[c % len(branch_priors)] if branch_priors else ""
            for ell in range(L):
                if time.time() > deadline:
                    stop_reason = "time_budget"
                    break
                ex = ctx.budget_exhausted()
                if ex:
                    stop_reason = ex
                    break
                out = creator.propose(pool=pool, step=ell, branch=c,
                                      branch_prior=prior, k=K, skills=skills)
                for cand in out["candidates"]:
                    if inspector is not None:
                        rev = inspector.review(cand)
                        n_reviewed += 1
                        cand.provenance["inspector"] = {
                            "verdict": rev["verdict"], "source": rev["source"],
                            "violations": [v["constraint"] for v in rev["violations"]],
                            "min_fix": [v.get("min_fix", "") for v in rev["violations"]],
                            "unverified": rev["unverified_assumptions"]}
                        if rev["verdict"] == "reject":
                            cand.status = "rejected"
                            n_rejected += 1
                        else:
                            n_admitted += 1
                    else:
                        n_admitted += 1 if cand.status != "rejected" else 0
                    archive.update(cand)
                # 刷新 pool (本步新增候选)
                pool = project(archive.all(), view=view, max_n=max_ctx,
                               shuffle=False) if len(archive) else pool
                rec = {"event": "step", "r": r, "c": c, "ell": ell,
                       "n_pool": len(pool), "n_new": len(out["candidates"]),
                       "thought": out["thought"][:200],
                       "actions": out["actions"], "prior": prior}
                audit.log(**rec)
                step_log.append(rec)
                # 早停
                best = archive.best(admissible_only=True)
                if best is not None and domain is not None and hasattr(domain, "target_met"):
                    if domain.target_met(best):
                        best_submitted = best
                        stop_reason = "target_met"
                        break
            if stop_reason in ("time_budget", "token_budget", "cost_budget", "target_met"):
                break
        if stop_reason in ("time_budget", "token_budget", "cost_budget", "target_met"):
            break

    if stop_reason == "completed":
        stop_reason = "loop_exhausted"

    ranked = archive.ranked(admissible_only=True) or archive.ranked()
    best = best_submitted or (ranked[0] if ranked else None)
    # 去重 (同 dsl 只留最优) 后再截断, 便于阅读与后续 Pareto 分析
    seen: Dict[str, Dict[str, Any]] = {}
    for c in ranked:
        seen.setdefault(c.dsl, c.to_dict())
    top = list(seen.values())[:8]

    result = {
        "task": task,
        "engine": "agent",
        "config": {"R": R, "C": C, "L": L, "K": K, "projector": view,
                   "inspector": inspector is not None,
                   "branch_priors": branch_priors},
        "best": best.to_dict() if best else None,
        "top": top,
        "n_candidates": len(archive),
        "n_reviewed": n_reviewed,
        "n_admitted": n_admitted,
        "n_rejected": n_rejected,
        # 对标文献口径: 入池候选通过全部约束的比率
        "admit_ratio": (n_admitted / n_reviewed) if n_reviewed else None,
        "stop_reason": stop_reason,
        "elapsed_s": round(time.time() - t_start, 2),
        "cost": ledger.to_dict(),
        "archive": archive_path,
        "traj": traj_path,
        "steps": len(step_log),
    }
    audit.log(event="finish", **{k: v for k, v in result.items()
                                 if k not in ("top", "config")})
    _dump_json(os.path.join(base_dir, "result.json"), result)
    return result


def _branch_priors(domain, cfg, C: int) -> List[str]:
    priors = list(getattr(cfg, "agent_branch_priors", None) or [])
    if not priors and domain is not None and hasattr(domain, "branch_priors"):
        priors = list(domain.branch_priors(cfg) or [])
    if not priors and C > 1:
        priors = [f"branch-{i}" for i in range(C)]
    return priors[:C] if priors else []


def _dump_json(path: str, obj: Any) -> None:
    import json
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
