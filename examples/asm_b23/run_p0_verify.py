"""示例: P0 判决点 —— archive 回灌 + 投影复现 + C1 奇偶性机器判定.

对应方案 §4 的 P0 验收 (全案第一个"值不值得继续"的判决点, **不调 LLM**):

1. 把既有 `outputs/asm/runs_*/*_final.json` 全量回灌进 append-only archive;
2. 用 Pareto 投影 + archive 查询直接复现历史结论 (无需再手写一次性脚本);
3. 对 C1 奇偶性探针, **机器判定** T1_23/T5_23/T6_23/T10_23 为偶对称,
   并自动标记 physics context 中那条错误断言。

判据: 若第 3 条不能自动跑出来, 说明约束层设计有问题 —— 不要进入 P1。

用法:
    python examples/asm_b23/run_p0_verify.py
    python examples/asm_b23/run_p0_verify.py --legacy ../../outputs/asm --out ./p0_report
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import resolve_dataset                               # noqa: E402
from lgf_agent import LGFConfig                                   # noqa: E402
from lgf_agent.projector import pareto_front, project, render     # noqa: E402
from lgf_agent.store import Archive                               # noqa: E402
from lgf_agent.domains.asm import ASMDomain                       # noqa: E402


def _dump(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    help="缺省: 打包的 64 点/工况精简样例")
    ap.add_argument("--legacy", default=os.path.join(os.path.dirname(REPO),
                                                     "outputs", "asm"),
                    help="既有 runs_* 产物目录 (会递归找 *_final.json)")
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "p0_report"))
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    report = {"steps": {}}

    # ---------- 步骤 3 (先做, 因为它不依赖任何历史产物) ----------
    ds_path = resolve_dataset(a.dataset, "g")
    print(f"[p0] dataset = {ds_path}")
    dom = ASMDomain.from_dataset(ds_path, variant="g")
    audit = dom.parity_audit()
    report["steps"]["C1_parity_audit"] = audit
    print("=== [步骤 3] C1 奇偶性机器判定 ===")
    print(f"  对照 (已知奇函数 yh): even_score = {audit['control_odd_yh']:+.3f}  "
          f"(应 ≈ -1, 证明探针有区分力)")
    for k, v in audit["channels"].items():
        print(f"  {k:10s} even_score = {v:+.3f}")
    print(f"  声明一致性: {audit['consistent']}")
    for name, rec in audit["claim"].items():
        flag = "OK " if rec["verified"] else "!! "
        print(f"  {flag}{name}: 声明={rec['declared']} 实测={rec['even_score']:+.3f}")

    bad = dom.known_bad_assertions()
    report["steps"]["known_bad_assertions"] = bad
    if bad:
        print("\n  [自动标记] 历史错误断言:")
        for r in bad:
            print(f"    - {r['source_file']}:{r['source_lines']}")
            print(f"      断言: {r['assertion']}")
            print(f"      裁决: {r['verdict']}  (证据: {r['machine_evidence']})")

    # ---------- 步骤 1: archive 回灌 ----------
    print("\n=== [步骤 1] 既有 runs_* 回灌 archive ===")
    files = []
    for pat in ("*_final.json", "summary_*.json"):
        files.extend(glob.glob(os.path.join(a.legacy, "**", pat), recursive=True))
    print(f"  发现 {len(files)} 个产物文件 (*_final.json / summary_*.json)")
    arch_path = os.path.join(a.out, "archive_replay.jsonl")
    if os.path.exists(arch_path):
        os.remove(arch_path)
    archive = Archive.from_legacy_runs(a.legacy, task="asm_replay",
                                       archive_path=arch_path,
                                       globs=["*_final.json", "summary_*.json"])
    report["steps"]["replay"] = {
        "n_final_json": len(files), "n_candidates": len(archive),
        "archive": arch_path,
        "events_tail": archive.events[-3:],
    }
    print(f"  回灌候选数: {len(archive)}; archive -> {arch_path}")

    # ---------- 步骤 2: Pareto 投影 ----------
    print("\n=== [步骤 2] 投影视图 (直接复现历史排序, 无需一次性脚本) ===")
    if len(archive):
        # 历史产物跨多个"变量集"(口径不同), 必须分组比较 —— 否则 0.0131 这种
        # 看似更好的 loss 其实来自 b23 之外的另一个目标。这正是 archive 的价值:
        # 分组信息 (var_names) 随候选一起落盘。
        groups: dict = {}
        for c in archive:
            key = tuple(c.var_names or ())
            groups.setdefault(key, []).append(c)
        proj = {"n_all": len(archive), "groups": {}}
        print(f"  共 {len(archive)} 条候选, 按变量集分为 {len(groups)} 组:")
        for key, cs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            front = pareto_front(cs)
            label = "|".join(key) if key else "(无变量集信息)"
            proj["groups"][label] = {
                "n": len(cs),
                "pareto": [{"dsl": c.dsl, "loss": c.loss, "nodes": c.n_nodes}
                           for c in front[:8]],
            }
            print(f"\n  ── 变量集 [{label}] : {len(cs)} 条, 前沿 {len(front)} 条")
            for c in front[:6]:
                print(f"     loss={c.loss:<11.5g} nodes={c.n_nodes:<3d} {c.dsl[:66]}")
        report["steps"]["projection"] = proj

        biggest = max(groups.values(), key=len)
        print("\n  注入 prompt 的有界视图 (最大组 · pareto · ≤8):")
        print("  " + render(biggest, view="pareto", max_n=8).replace("\n", "\n  "))
    else:
        report["steps"]["projection"] = {"n_all": 0, "note": "未找到历史产物"}
        print("  (未找到历史产物, 跳过。请用 --legacy 指定目录)")

    _dump(os.path.join(a.out, "p0_report.json"), report)
    print(f"\n[done] 报告 -> {os.path.join(a.out, 'p0_report.json')}")


if __name__ == "__main__":
    main()
