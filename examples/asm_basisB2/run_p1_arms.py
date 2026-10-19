"""P1: 三臂同预算对照 —— legacy vs agent(-Inspector) vs agent(+Inspector).

方案 §5.1 的核心实验协议。三臂跑**完全相同的任务** (basisB2 口径, e 基 a₂
系数 SR), 唯一变量是引擎:

    legacy            core.pipeline_lgf.run_lgf (原管线, kernel 原样)
    agent_no_insp     run_agent, agent_inspector=False
    agent_insp        run_agent, agent_inspector=True

同预算硬约束: 三臂共用 seed / max_edges / max_params / **time_budget_s**
(两引擎内部都尊重该字段, 到点即停并写 stop_reason)。

用法 (heuristic 协议验证, 无需 key):
    python examples/asm_basisB2/run_p1_arms.py --budget 60
真 LLM (key 在仓库 .env):
    python examples/asm_basisB2/run_p1_arms.py --budget 600 --backend openai
输出: run/basisb2_p1_arms/{arm}/... + p1_arms.json 汇总对比表。
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import resolve_datasets, default_out                 # noqa: E402
from lgf_agent import LGFConfig                                   # noqa: E402
from lgf_agent.domains.asm.basisb2 import (                       # noqa: E402
    BasisB2Domain, load_channels, stage_base)


def load_env_file(path: str) -> bool:
    if not os.path.exists(path):
        return False
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    return bool(os.environ.get("LGF_LLM_API_KEY"))


def _best_loss_from_arm(out: dict) -> float:
    """两引擎返回结构不同 → 统一取最好 loss。"""
    for path in ("best", ):
        b = out.get(path) or {}
        if isinstance(b, dict) and b.get("loss") is not None:
            return float(b["loss"])
    if out.get("top"):
        top = out["top"]
        for cand in top:
            loss = getattr(cand, "loss", None)
            if loss is not None and loss != float("inf"):
                return float(loss)
            if isinstance(cand, dict) and cand.get("loss") is not None:
                return float(cand["loss"])
    return float("inf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--variant", default="e", choices=["v", "g", "e"])
    ap.add_argument("--target", type=int, default=2)
    ap.add_argument("--budget", type=float, default=60.0,
                    help="每臂 time_budget_s (三臂相同)")
    ap.add_argument("--L", type=int, default=6, help="agent_L (heuristic 冒烟用小值)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default=None,
                    help="逗号分隔多种子 (P2), 如 --seeds 0,1,2; 给定时覆盖 --seed")
    ap.add_argument("--backend", default="heuristic",
                    choices=["heuristic", "openai"])
    ap.add_argument("--arms", default="legacy,agent_no_insp,agent_insp",
                    help="逗号分隔, 子集重跑 (如只重跑 inspector 臂: --arms agent_insp)")
    ap.add_argument("--out", default=default_out("basisb2_p1_arms"))
    a = ap.parse_args()
    arm_sel = [s.strip() for s in a.arms.split(",") if s.strip()]
    seeds = ([int(s) for s in a.seeds.split(",") if s.strip()]
             if a.seeds else [a.seed])

    have_key = load_env_file(os.path.join(REPO, ".env")) or \
        bool(os.environ.get("LGF_LLM_API_KEY"))
    if a.backend == "openai" and not have_key:
        print("[p1] --backend openai 但未找到 key, 退回 heuristic。")
        a.backend = "heuristic"

    # ---- 域与任务 (三臂完全相同) ----
    paths = resolve_datasets(a.data)
    cases = load_channels(a.variant, paths, cnorm="percase")
    base = stage_base(cases)
    dom = BasisB2Domain(base, target_index=a.target - 1, variant=a.variant)
    scases, vn = dom.build_cases(LGFConfig())
    print(f"[p1] 任务 = basisB2 {a.variant} 基 a_{a.target} ({dom.channel_name}), "
          f"{len(scases)} 工况, 变量 {len(vn)} 维, 预算 {a.budget:.0f}s/臂, "
          f"backend={a.backend}")

    def make_cfg(seed: int, **kw) -> LGFConfig:
        c = LGFConfig(seed=seed, max_edges=24, max_params=12,
                      time_budget_s=a.budget, **kw)
        return c

    from lgf_agent.core import llm_ops
    client = llm_ops.make_client(a.backend, url=os.environ.get("LGF_LLM_URL"),
                                 model=os.environ.get("LGF_LLM_MODEL", "glm-5.3"),
                                 api_key=os.environ.get("LGF_LLM_API_KEY"),
                                 max_tokens=16384, timeout=300.0,
                                 extra_body={"thinking": {"type": "enabled"},
                                             "reasoning_effort": "low"})
    insp_client = client                      # P4 才换独立底座

    if a.backend == "openai":
        # 连通性预检: 限额耗尽时别启动几十分钟的跑批 (空 propose 会产出垃圾臂)
        try:
            client.complete('回复一个词: ok', n=1)
            print("[p1] LLM 连通性预检 OK")
        except Exception as e:                                # noqa: BLE001
            print(f"[p1] LLM 预检失败, 放弃启动: {str(e)[:200]}")
            return

    selected = []
    if "legacy" in arm_sel:
        selected.append("legacy")
    if "agent_no_insp" in arm_sel:
        selected.append("agent_no_insp")
    if "agent_insp" in arm_sel:
        selected.append("agent_insp")

    from lgf_agent.core.pipeline_lgf import run_lgf
    from lgf_agent.agent.scheduler import run_agent

    all_results = {}                          # seed -> {arm -> record}
    for seed in seeds:
        out_root = a.out if len(seeds) == 1 else os.path.join(a.out, f"seed{seed}")
        arms = {}

        # ---- 臂 1: legacy ----
        if "legacy" in selected:
            cfg = make_cfg(seed)
            t0 = time.time()
            print(f"\n[seed {seed} | arm legacy] ...", flush=True)
            out = run_lgf([dict(c) for c in scases], list(vn), cfg,
                          os.path.join(out_root, "legacy"), client, name="legacy")
            arms["legacy"] = dict(best_loss=_best_loss_from_arm(out),
                                  elapsed_s=round(time.time() - t0, 2),
                                  n_final=len(out.get("top_final") or []),
                                  dir=os.path.join(out_root, "legacy"))

        # ---- 臂 2/3: agent ±Inspector ----
        for name, inspector in [("agent_no_insp", False), ("agent_insp", True)]:
            if name not in selected:
                continue
            cfg = make_cfg(seed, agent_mode=True, agent_inspector=inspector,
                           agent_L=a.L)
            print(f"\n[seed {seed} | arm {name}] (L={a.L}) ...", flush=True)
            out = run_agent(dom, cfg, llm_client=client,
                            inspector_client=insp_client if inspector else None,
                            out_dir=os.path.join(out_root, name), task_name=name)
            arms[name] = dict(best_loss=_best_loss_from_arm(out),
                              elapsed_s=out.get("elapsed_s"),
                              n_candidates=out.get("n_candidates"),
                              admit_ratio=out.get("admit_ratio"),
                              n_deferred=out.get("n_deferred"),
                              stop_reason=out.get("stop_reason"),
                              cost=out.get("cost"),
                              dir=os.path.join(out_root, name))

        # ---- 单种子汇总 ----
        print(f"\n=== P1 三臂对比 (seed={seed}, 同任务/同预算) ===")
        print(f"{'arm':<16} {'best_loss':>11} {'耗时s':>8} {'备注':<28}")
        for name, r in arms.items():
            note = (f"stop={r.get('stop_reason')}, admit={r.get('admit_ratio')}"
                    if name != "legacy" else
                    f"top_final={r['n_final']}")
            loss = r["best_loss"]
            print(f"{name:<16} {loss if loss != float('inf') else 'inf':>11.4g} "
                  f"{r['elapsed_s']:>8.1f} {note:<28}")

        all_results[seed] = arms
        os.makedirs(out_root, exist_ok=True)
        with open(os.path.join(out_root, "p1_arms.json"), "w",
                  encoding="utf-8") as f:
            json.dump(dict(task=dict(variant=a.variant, target=a.target,
                                     seed=seed, budget_s=a.budget,
                                     backend=a.backend),
                           arms=arms), f, ensure_ascii=False, indent=2,
                      default=str)

    # ---- 多种子聚合 (P2) ----
    if len(seeds) > 1:
        print(f"\n=== 多种子聚合 ({len(seeds)} seeds: {seeds}) ===")
        print(f"{'arm':<16} {'median':>10} {'mean':>10} {'min':>10} {'max':>10}")
        agg = {}
        for name in selected:
            losses = [all_results[s][name]["best_loss"] for s in seeds
                      if name in all_results.get(s, {})]
            losses = [l for l in losses if l != float("inf")]
            if losses:
                losses.sort()
                n = len(losses)
                med = (losses[n // 2] if n % 2 else
                       0.5 * (losses[n // 2 - 1] + losses[n // 2]))
                agg[name] = dict(median=med, mean=sum(losses) / n,
                                 min=losses[0], max=losses[-1], n=n)
                print(f"{name:<16} {med:>10.4g} {sum(losses)/n:>10.4g} "
                      f"{losses[0]:>10.4g} {losses[-1]:>10.4g}")
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "p1_arms_multiseed.json"), "w",
                  encoding="utf-8") as f:
            json.dump(dict(task=dict(variant=a.variant, target=a.target,
                                     seeds=seeds, budget_s=a.budget,
                                     backend=a.backend),
                           per_seed={str(s): all_results[s] for s in seeds},
                           aggregate=agg), f, ensure_ascii=False, indent=2,
                      default=str)
        print(f"\n[done] -> {os.path.join(a.out, 'p1_arms_multiseed.json')}")
    else:
        print(f"\n[done] -> {os.path.join(a.out, 'p1_arms.json')}")


if __name__ == "__main__":
    main()
