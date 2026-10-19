"""示例: 回放历史冠军 — 独立复算终口径 joint refit (对账 0.0576).

从上层工程 `outputs/asm/runs_basisB2/<group>/result.json` 读冠军 DSL (公式形状),
在 LGF-Agent 包内**重做**联合重拟合 (refit.joint_refit), 与记录值对账:

    e_lgf_mg_big: 0.7918 (中间) → 0.0576 (refit)   ⭐ 总冠军
    g_lgf_mg_big: 0.376  (中间) → 0.1238
    m_lgf_go_big: 34.03  (中间) → 0.0722           ⭐ m 池 refit 治愈的活标本
    v_lgf_go:     7.668  (中间) → 2.231            (v 基不可行, 作对照)

这证明 LGF-Agent 是**自包含**的: 终口径数字可以独立复算, 不是引用报告。
需要上层工程的 runs_basisB2 与 v/g/e 完整数据集 (demo 数据会带来 ~2-5% 偏差,
因为 64 点 ≠ 129 点; 对账请用完整数据)。

用法:
    python examples/asm_basisB2/run_refit_replay.py --data <上层 outputs/asm> \
        --groups e_lgf_mg_big,g_lgf_mg_big,m_lgf_go_big
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import resolve_datasets, resolve_runs, default_out      # noqa: E402
from lgf_agent.domains.asm.refit import replay_run_refit             # noqa: E402

DEFAULT_GROUPS = "e_lgf_mg_big,g_lgf_mg_big,m_lgf_go_big,v_lgf_go"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="含 dataset_trueasm_{v,g,e}.npz 的目录 (对账请用完整数据)")
    ap.add_argument("--runs", default=None, help="runs_basisB2 目录")
    ap.add_argument("--groups", default=DEFAULT_GROUPS)
    ap.add_argument("--out", default=default_out("basisb2_refit_replay"))
    a = ap.parse_args()

    runs = resolve_runs(a.runs)
    if not runs:
        raise SystemExit("找不到 runs_basisB2 目录 (上层工程产物), 请用 --runs 指定")
    paths = resolve_datasets(a.data)
    print(f"[replay] runs = {runs}")
    print(f"[replay] data = { {k: os.path.basename(v) for k, v in paths.items()} }")
    demo = "_demo" in next(iter(paths.values()))
    if demo:
        print("[replay] ⚠️ 用的是精简 demo 数据 (64 点), 复算值会偏离记录 "
              "(129 点) — 对账请用完整数据集")

    rows, ok_all = [], True
    for grp in [g.strip() for g in a.groups.split(",") if g.strip()]:
        print(f"\n=== {grp} ===")
        try:
            r = replay_run_refit(runs, grp, paths)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue
        rec_i, rec_r = r["recorded"]["intermediate"], r["recorded"]["refit"]
        rep_i, rep_r = r["replayed"]["intermediate"], r["replayed"]["refit"]
        def _rel(rep, rec):
            if rep is None or rec in (None, 0):
                return float("nan")
            return abs(rep - rec) / abs(rec)
        ri, rr = _rel(rep_i, rec_i), _rel(rep_r, rec_r)
        ok = (rr < 0.05) if (rep_r is not None and rec_r) else False
        ok_all = ok_all and ok
        print(f"  中间口径: 记录 {rec_i:.4f} | 复算 {rep_i:.4f} (rel {ri:.2%})")
        print(f"  终口径  : 记录 {rec_r:.4f} | 复算 {rep_r:.4f} (rel {rr:.2%}) "
              f"{'✅' if ok else '⚠️' if rr < 0.15 else '❌'}")
        if r["refit_detail"]:
            print("  per-case: " + " ".join(
                f"{k[2:]}:{v:.3f}" for k, v in
                r["refit_detail"]["per_case"].items()))
        rows.append(dict(group=grp, variant=r["variant"], J=r["J"],
                         recorded=r["recorded"], replayed=r["replayed"],
                         rel_refit=rr, within_5pct=ok))

    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "refit_replay.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "all_within_5pct": ok_all}, f,
                  ensure_ascii=False, indent=2, default=str)
    print(f"\n[done] {'全部 ✅ (终口径 5% 内复算一致)' if ok_all else '存在偏离, 见上表'}"
          f" -> {out}")


if __name__ == "__main__":
    main()
