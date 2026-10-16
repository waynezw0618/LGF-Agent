"""示例: 离线 (无 LLM) 跑通 LGF-Agent 的 R-C-L-K 循环.

用途:
- CI / 冒烟: 验证 archive / 投影 / 数值探针 / 调度器全链路可跑;
- legacy 对照: 与本目录的 `run_legacy_compare.py` 同预算对比。

用法:
    python examples/asm_b23/run_offline_demo.py --dataset ../../outputs/asm/dataset_trueasm_g.npz
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import resolve_dataset                            # noqa: E402
from lgf_agent import LGFConfig                                # noqa: E402
from lgf_agent.agent.scheduler import run_agent                # noqa: E402
from lgf_agent.domains.asm import ASMDomain                    # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    help="缺省: 打包的 64 点/工况精简样例")
    ap.add_argument("--variant", default="g", choices=["v", "g", "e"])
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "agent_demo"))
    ap.add_argument("--budget", type=float, default=180.0)
    ap.add_argument("--R", type=int, default=1)
    ap.add_argument("--C", type=int, default=2)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--projector", default="pareto",
                    choices=["pareto", "topk", "single_best"])
    ap.add_argument("--no-inspector", action="store_true")
    a = ap.parse_args()

    ds = resolve_dataset(a.dataset, a.variant)
    print(f"[demo] dataset = {ds}")
    dom = ASMDomain.from_dataset(ds, variant=a.variant)
    cfg = LGFConfig(
        population=10, generations=3, time_budget_s=a.budget,
        gp_starts=4, gp_fit_budget_s=6.0, final_starts=6, final_maxiter=120,
        max_edges=15, max_params=6, seed=0,
        agent_mode=True, agent_R=a.R, agent_C=a.C, agent_L=a.L, agent_K=a.K,
        agent_projector=a.projector, agent_inspector=not a.no_inspector,
    )
    res = run_agent(dom, cfg, out_dir=a.out, task_name="asm_offline_demo")

    print(f"[demo] stop={res['stop_reason']} steps={res['steps']} "
          f"candidates={res['n_candidates']}")
    print(f"[demo] reviewed={res['n_reviewed']} admitted={res['n_admitted']} "
          f"rejected={res['n_rejected']} admit_ratio={res['admit_ratio']}")
    best = res.get("best") or {}
    print(f"[demo] BEST loss={best.get('loss')} :: {best.get('dsl')}")
    print(f"[demo] archive -> {res['archive']}")
    print(f"[demo] traj    -> {res['traj']}")
    print(f"[demo] result  -> {os.path.join(a.out, 'agent', 'asm_offline_demo', 'result.json')}")


if __name__ == "__main__":
    main()
