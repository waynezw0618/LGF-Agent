"""示例: 方案B v2 (多通道系数口径) 离线跑通 — stage base + 单通道 SR demo.

不调 LLM (heuristic 通道), 验证:
1. `basisb2.load_channels` + `stage_base`: 结构化联合解 + β 靶标 (J=4);
2. `BasisB2Domain(target_index=j)`: 一条系数 a_j(Θ) 的标准 LGF-Agent 任务
   (13 动作全链路, R-C-L-K 离线跑通);
3. 中间口径重组 + (可选小规模) joint refit 演示两级口径差异。

用法:
    python examples/asm_basisB2/run_offline_demo.py                 # e 基, target a1
    python examples/asm_basisB2/run_offline_demo.py --variant g --target 2
    python examples/asm_basisB2/run_offline_demo.py --data <上层 outputs/asm>
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                                # noqa: E402
from _common import resolve_datasets, default_out                 # noqa: E402
from lgf_agent import LGFConfig                                   # noqa: E402
from lgf_agent.domains.asm.basisb2 import (                       # noqa: E402
    BasisB2Domain, load_channels, stage_base, reconstruct)
from lgf_agent.core import dsl as _dsl                            # noqa: E402
from lgf_agent.core.paramfit import fit_joint                     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="含 dataset_trueasm_{v,g,e}.npz 的目录")
    ap.add_argument("--variant", default="e", choices=["v", "g", "e"])
    ap.add_argument("--target", type=int, default=1, help="拟合第几条系数 (1-based)")
    ap.add_argument("--out", default=default_out("basisb2_demo"))
    ap.add_argument("--budget", type=float, default=120.0)
    a = ap.parse_args()

    paths = resolve_datasets(a.data)
    print(f"[demo] data = {paths[a.variant]}")

    # ---- 1) stage base: 结构化联合解 + β 靶标 ----
    cases = load_channels(a.variant, paths, cnorm="percase")
    base = stage_base(cases)
    print(f"[base] J={base['J']} K={base['K']} (48 参数联合解口径) "
          f"loss_vw={base['loss_vw']:.4f} worst={base['worst']:.4f}")
    print(f"[base] beta_rms = " + " ".join(
        f"{v:.3g}" for v in base["beta_rms"]))

    # ---- 2) 单通道系数任务: BasisB2Domain + 离线 R-C-L-K ----
    dom = BasisB2Domain(base, target_index=a.target - 1, variant=a.variant)
    print(f"\n[task] a_{a.target} (通道 {dom.channel_name}), "
          f"scale={dom.scale:.4g}")

    from lgf_agent.agent.scheduler import run_agent
    cfg = LGFConfig(
        population=10, generations=3, time_budget_s=a.budget,
        gp_starts=4, gp_fit_budget_s=6.0, final_starts=6, final_maxiter=120,
        max_edges=15, max_params=6, seed=0,
        agent_mode=True, agent_R=1, agent_C=2, agent_L=3, agent_K=1,
        agent_projector="pareto", agent_inspector=True)
    res = run_agent(dom, cfg, out_dir=a.out, task_name="basisb2_offline_demo")
    print(f"[agent] stop={res['stop_reason']} candidates={res['n_candidates']} "
          f"admit_ratio={res['admit_ratio']:.3f}")
    best = res.get("best") or {}
    print(f"[agent] BEST (系数 NMSE) = {best.get('loss')} :: {best.get('dsl')}")

    # ---- 3) 两级口径演示: 用 4 条结构种子做 SR 形状 → 中间重组 vs refit ----
    print("\n[two-stage] 用 domain.seed_skeletons 做 4 通道形状 (离线种子级):")
    dsls, scales, thetas = [], [], []
    for j in range(base["J"]):
        dj = BasisB2Domain(base, target_index=j, variant=a.variant)
        dj_cases, _ = dj.build_cases(cfg)
        seed = dj.seed_skeletons(branch=j % 4)[0]
        try:
            g = _dsl.parse_dsl(seed, list(dj.var_names))
        except Exception as e:                                  # noqa: BLE001
            print(f"  a{j+1}: 种子解析失败 {e!r}")
            continue
        fr = fit_joint(g, dj_cases, n_starts=4, maxiter=80,
                       time_budget_s=8.0, seed=0)
        dsls.append(seed)
        scales.append(dj.scale)
        thetas.append([float(v) for v in fr.theta])
        print(f"  a{j+1} ({dj.channel_name}): 系数 NMSE={fr.loss:.3e} :: {seed[:58]}")
    if dsls:
        from lgf_agent.domains.asm.refit import (joint_refit,
                                                 refit_intermediate)
        inter = refit_intermediate(cases, dsls, scales, thetas)
        print(f"  中间口径重组 loss_vw = {inter['loss_vw']:.4f} "
              f"(worst {inter['worst']:.4f})")
        refit = joint_refit(cases, dsls, scales, thetas, verbose=False)
        if refit:
            print(f"  终口径 joint refit loss_vw = {refit['loss_vw']:.4f} "
                  f"(worst {refit['worst']:.4f}, n_par={refit['n_params']}) "
                  f"← 两级口径差 = "
                  f"{inter['loss_vw'] / max(refit['loss_vw'], 1e-12):.1f}×")
        else:
            print("  终口径 refit 失败 (LM 不收敛, 见 refit.py verbose)")


if __name__ == "__main__":
    main()
