"""示例: 汇总上层 runs_basisB2 → 复现报告 §3.1 完整矩阵 (纯读取, 不复算).

读 runs_basisB2/*/result.json 的 loss_vw (中间) 与 refit.loss_vw (终口径),
按变体分组打印 — 与 BasisB2_full_matrix_report.md §3.1 的表逐行对照。
另调用 channel_usage/vars_used 做通道归因 (0.2.0 的能力在新口径下的应用)。

用法:
    python examples/asm_basisB2/run_matrix_replay.py
    python examples/asm_basisB2/run_matrix_replay.py --runs <runs_basisB2>
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

from _common import resolve_runs, default_out                    # noqa: E402
from lgf_agent.domains.asm.refit import collect_matrix           # noqa: E402
from lgf_agent.domains.asm.basisb2 import NAMED_CHANNELS         # noqa: E402

# 报告 §3.1 的行 (方法名 → 显示名)
REPORT_ROWS = [
    ("pysr", "PySR full"),
    ("pysr_ms12", "PySR ms12"),
    ("lgf_mg", "LGF-mg legacy"),
    ("lgf_go", "LGF-go legacy"),
    ("lgf_mg_v2", "LGF-mg v2"),
    ("lgf_go_v2", "LGF-go v2"),
    ("lgf_go_nophys", "LGF-go nophys"),
    ("lgf_mg_big", "LGF-mg big"),
    ("lgf_go_big", "LGF-go big"),
]
REF = {"PySR full": {"g": (0.357, 0.128), "e": (0.963, 0.377), "m": (46.10, 1.173)},
       "LGF-mg big": {"g": (0.376, 0.124), "e": (0.792, 0.0576), "m": (61.12, 0.926)},
       "LGF-go big": {"g": (0.550, 0.140), "e": (0.856, 0.134), "m": (34.03, 0.0722)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None)
    ap.add_argument("--out", default=default_out("basisb2_matrix"))
    a = ap.parse_args()
    runs = resolve_runs(a.runs)
    if not runs:
        raise SystemExit("找不到 runs_basisB2, 请用 --runs 指定")
    rows = collect_matrix(runs)
    print(f"[matrix] {len(rows)} 组 <- {runs}\n")

    # 按 (variant, method-tag) 组织
    by = {}
    for r in rows:
        by.setdefault(r["variant"], {})[r["group"]] = r

    print(f"{'组':<18}" + "".join(f"{v+' 中间':>12}{v+' 终值':>12}" for v in "vgem"))
    print("-" * 114)
    for meth, disp in REPORT_ROWS:
        cells = []
        for v in "vgem":
            g = by.get(v, {})
            # group 命名: {variant}_{method}{sfx}, 无 tag = legacy
            hit = None
            for grp, r in g.items():
                if grp == f"{v}_{meth}":
                    hit = r
                    break
            if hit:
                cells.append(f"{hit['loss_mid']:>12.3f}"
                             f"{(hit['loss_refit'] or float('nan')):>12.3f}")
            else:
                cells.append(f"{'—':>12}{'—':>12}")
        print(f"{disp:<18}" + "".join(cells))

    # 对账: 与报告 §3.1 的锚点 (PySR 组 group 名可能带 _s{seed} 后缀)
    print("\n锚点对账 (报告 §3.1):")
    for disp, vs in REF.items():
        for v, (mid, fin) in vs.items():
            hit = None
            for grp, r in by.get(v, {}).items():
                if grp == f"{v}_{disp.lower().replace(' ', '_').replace('-', '_')}" \
                        or grp.startswith(f"{v}_{disp.lower().replace(' ', '_').replace('-', '_')}_s"):
                    hit = r
                    break
            if hit:
                got = hit["loss_refit"]
                flag = "✅" if (got is not None and abs(got - fin) / fin < 0.02) else "❌"
                print(f"  {disp} {v}: 报告 {fin} | 实读 {got} {flag}")
            else:
                print(f"  {disp} {v}: (本机无此组)")

    # 冠军公式通道归因 (0.2.0 能力在新口径的应用)
    print("\n冠军组 DSL 引用变量 (通道归因, 公式口径):")
    from lgf_agent import analysis
    for grp in ("e_lgf_mg_big", "g_lgf_mg_big", "m_lgf_go_big"):
        hit = None
        for r in rows:
            if r["group"] == grp:
                hit = r
                break
        if hit is None:
            continue
        run = json.load(open(os.path.join(runs, grp, "result.json"),
                             encoding="utf-8"))
        print(f"  {grp} (J={run['J']}):")
        for t in run["targets"]:
            u = analysis.candidate_channel_usage(t["dsl"],
                                                 ["eta2", "zeta2", "s2o2",
                                                  "ret", "roloc", "yw"])
            used = [v for v in ("eta2", "zeta2", "s2o2", "ret", "roloc", "yw")
                    if v in t["dsl"]]
            print(f"    {t['target']} ({t['dsl'][:52]}...): 用到 {used}")

    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "matrix.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
