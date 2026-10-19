"""全 4 通道 agent SR 化 → 终口径联合抛光 (对齐 0.0576 协议).

故事闭环: P1/P2 已证 agent 发现的 a2 在同帧替换中优于人工 big 轮冠军
(0.0726 vs 0.1992)。本脚本把 4 条通道全部换成 agent 自动发现的公式
(a1/a3/a4 来自 run/basisb2_full_agent/a*/, a2 来自 P1), 用 refit.joint_refit
做与历史 SOTA e_lgf_mg_big (0.0576) **完全相同协议**的终口径联合抛光。

判读:
    joint_refit(4 条 agent 公式)  vs  0.0576 (4 条人工轮公式, 55 参)
    —— 若 ≤0.0576: agent 全流程替代人工调参的最强证据;
       若 1.0-1.5x: 同级, 用参数数与预算成本补足叙事。

用法:
    python examples/asm_basisB2/refit_full_agent.py
输出: run/basisb2_full_agent/full_agent_refit.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import resolve_datasets, default_out                 # noqa: E402
from lgf_agent.domains.asm.basisb2 import (                       # noqa: E402
    load_channels, stage_base)
from lgf_agent.domains.asm.refit import joint_refit, refit_intermediate  # noqa: E402


def _clean_dsl(s: str) -> str:
    """LLM 偶尔在 DSL 前带 'a_j = ' 标签; compile_expr 只吃纯表达式。"""
    if s and "=" in s and not s.strip().startswith("("):
        head = s.split("=", 1)[0].strip()
        if head and head.replace("_", "").isalnum():   # 形如 a1 / a_1 的标签
            s = s.split("=", 1)[1].strip()
    return s


def _top1(pattern: str):
    """从 run 目录抓 top-1 冠军。返回 (dsl, theta) 或 None。"""
    for p in sorted(glob.glob(pattern, recursive=True)):
        try:
            r = json.load(open(p, encoding="utf-8"))
        except Exception:                                      # noqa: BLE001
            continue
        top = r.get("top") or []
        if top:
            c = top[0]
            return _clean_dsl(str(c["dsl"])), c.get("theta")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--variant", default="e", choices=["v", "g", "e"])
    ap.add_argument("--full-dir", default=default_out("basisb2_full_agent"))
    ap.add_argument("--p1-dir", default=default_out("basisb2_p1_arms"))
    ap.add_argument("--max-nfev", type=int, default=20000)
    a = ap.parse_args()

    # ---- 数据与 scales (与 replay_run_refit 同口径) ----
    paths = resolve_datasets(a.data)
    cases = load_channels(a.variant, paths, cnorm="percase")
    base = stage_base(cases)
    scales = [float(s) for s in base["beta_rms"]]

    # ---- 4 条冠军: a1/a3/a4 来自 full_agent 目录, a2 来自 P1 ----
    srcs = [
        ("a1", os.path.join(a.full_dir, "a1", "**", "result.json")),
        ("a2", os.path.join(a.p1_dir, "agent_no_insp", "**", "result.json")),
        ("a3", os.path.join(a.full_dir, "a3", "**", "result.json")),
        ("a4", os.path.join(a.full_dir, "a4", "**", "result.json")),
    ]
    dsls, thetas, missing = [], [], []
    for name, pat in srcs:
        got = _top1(pat)
        if got is None:
            missing.append(name)
            dsls.append("0")                 # 占位 (公式编译为常 0)
            thetas.append(None)
        else:
            dsls.append(got[0])
            thetas.append(got[1])
    print("[full-agent] 4 通道冠军来源:")
    for (name, _), dsl, th in zip(srcs, dsls, thetas):
        tag = "缺失(置零通道)" if dsl == "0" else f"{len(th or [])} 参"
        print(f"  {name}: [{tag}] {dsl[:96]}")
    if missing:
        print(f"[warn] 通道 {missing} 无结果 —— 本轮只做可用通道的联合 refit, "
              "结论将注明通道缺失。")

    # ---- 中间口径 (SR 原常数直接重组) ----
    inter = refit_intermediate(cases, dsls, scales, thetas)
    print(f"\n[inter] 中间口径 loss_vw = {inter['loss_vw']:.4f}  "
          f"worst = {inter.get('worst', float('nan')):.4f}")

    # ---- 终口径联合抛光 (与 e_lgf_mg_big 0.0576 同协议) ----
    out = joint_refit(cases, dsls, scales, thetas, max_nfev=a.max_nfev)
    if out is None:
        print("[refit] 联合 LM 失败!")
        return
    n_par = out.get("n_params", sum(len(t or []) for t in thetas))
    print(f"[final] 终口径 loss_vw = {out['loss_vw']:.4f}  "
          f"worst = {out.get('worst', float('nan')):.4f}  "
          f"(~{n_par} 参数)")
    ratio = out["loss_vw"] / 0.0576
    print(f"[vs-SOTA] e_lgf_mg_big = 0.0576 (55 参, 人工大形状轮)  →  "
          f"agent 全通道 = {out['loss_vw']:.4f}  "
          f"({'✅ 优于 SOTA' if ratio < 1 else f'{ratio:.2f}x SOTA'})")

    os.makedirs(a.full_dir, exist_ok=True)
    with open(os.path.join(a.full_dir, "full_agent_refit.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(variant=a.variant, missing_channels=missing,
                       dsls=dsls, scales=scales,
                       intermediate=inter, final=out),
                  f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[done] -> {os.path.join(a.full_dir, 'full_agent_refit.json')}")


if __name__ == "__main__":
    main()
