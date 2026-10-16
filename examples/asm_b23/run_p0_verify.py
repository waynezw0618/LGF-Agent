"""示例: P0 判决点 —— archive 回灌 + 投影复现 + C1 奇偶性机器判定 + 通道归因.

对应方案 §4 的 P0 验收 (全案第一个"值不值得继续"的判决点, **不调 LLM**):

1. 把既有 `outputs/asm/runs_*/*_final.json` 全量回灌进 append-only archive;
2. 用 Pareto 投影 + archive 查询直接复现历史结论 (无需再手写一次性脚本);
3. 对 C1 奇偶性探针, **机器判定** T1_23/T5_23/T6_23/T10_23 为偶对称,
   并自动标记 physics context 中那条错误断言;
4. 通道归因: 从 archive 直接回答 "最优公式到底有没有用到张量基通道",
   并用已知真值的合成算例校准数值分解 (`leave_one_channel_out`)。
   这一条是历史上 `_tmp_t23_contrib.py` 一次性脚本的正式替代品。

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

import numpy as np                                                # noqa: E402
from _common import resolve_dataset, default_out                  # noqa: E402
from lgf_agent import LGFConfig                                   # noqa: E402
from lgf_agent import analysis                                    # noqa: E402
from lgf_agent.projector import pareto_front, project, render     # noqa: E402
from lgf_agent.store import Archive                               # noqa: E402
from lgf_agent.domains.asm import ASMDomain                       # noqa: E402


def _dump(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def _calibration_cases() -> list:
    """已知真值的合成算例 (用于校准 `leave_one_channel_out` 的数值分解).

    构造精确关系 ``y = 2.0·sqrt(roloc) + 0.5·T1`` (逐工况同系数), 于是:
      - 通道 T1 的 contribution 恰为 ``0.5·T1``  → rms_share 有解析值可比;
      - 置零 T1 后的 loss 也有解析值。
    另加一路纯标量工况, 用来验证 "零通道" 分支不会被误触发。
    """
    rng = np.random.default_rng(0)
    cases = []
    for i in range(3):
        n = 80
        roloc = rng.uniform(1.0, 60.0, n)
        t1 = rng.normal(0.0, 1.0, n)
        y = 2.0 * np.sqrt(roloc) + 0.5 * t1
        cases.append({"X": np.column_stack([roloc, t1]), "y": y,
                      "name": f"synth{i}"})
    return cases


def _step4_channel_attribution(archive: Archive, report: dict) -> None:
    print("\n=== [步骤 4] 通道归因 (历史上靠 _tmp_t23_contrib.py 一次性脚本回答的问题) ===")

    # ---- 4a. 结构查询: 零依赖, 只读 archive 的 var_names ----
    usage = archive.channel_usage()
    report["steps"]["channel_usage"] = usage
    print(f"  回灌 {usage['n_candidates']} 条候选: "
          f"公式口径 {usage['n_with_tensor_channel']} 条用到张量通道, "
          f"{usage['n_scalar_only']} 条纯标量 ({usage['frac_scalar_only']:.0%}); "
          f"池口径 {usage['n_pool_has_channel']} 条 (解析失败 {usage['n_parse_failed']} 条)")
    if usage["channels"]:
        print("  各通道被公式引用次数 (括号内为池中出现条数):")
        for ch, rec in usage["channels"].items():
            print(f"    {ch:9s} {rec['n_candidates']:4d} 条 "
                  f"({rec['n_pool_candidates']})")
    else:
        print("  (回灌记录里没有通道命名信息 —— 见下方 4b 的分组说明)")
    if usage["n_parse_failed"]:
        print(f"  注: {usage['n_parse_failed']} 条历史记录的 dsl 无法按记录的 var_names 解析 "
              f"(命名不全), 这些条目按**池口径**归类 —— 不计入上面的公式引用数。")
    if usage["best_scalar_only"]:
        print("  纯标量公式中最好的 5 条:")
        for r in usage["best_scalar_only"][:5]:
            print(f"    loss={r['loss']:<11.5g} id={r['id']} "
                  f"{(r['dsl'] or '')[:62]}")

    # ---- 4b. 逐候选判定: 在 "变量池含张量通道" 的最大分组内, 前排公式是否用了通道 ----
    # 这是历史争议的原点: T23-full 的 m 池里放了 10 个通道, 但前排公式到底用没用?
    groups: dict = {}
    for c in archive:
        groups.setdefault(tuple(c.var_names or ()), []).append(c)
    rows = []
    if groups:
        with_ch = [(k, v) for k, v in groups.items() if analysis.has_tensor_channel(k)]
        key, cs = max(with_ch or list(groups.items()), key=lambda kv: len(kv[1]))
        ranked = sorted([c for c in cs if c.loss is not None and c.loss < float("inf")],
                        key=lambda c: c.loss)[:8]
        n_ch = len(analysis.tensor_channels_used(key))
        print(f"\n  含通道变量集的分组共 {len(with_ch)} 个; "
              f"最大者: 变量池 {len(key)} 维 (其中 {n_ch} 个通道), {len(cs)} 条候选")
        print("  该组按 loss 排序的前 8 名 —— 公式里到底有没有通道:")
        for i, c in enumerate(ranked, 1):
            chs = analysis.candidate_channel_usage(c.dsl, c.var_names)["channels"]
            tag = "/".join(chs) if chs else "零通道 (纯标量式)"
            print(f"    #{i} loss={c.loss:<11.5g} 通道={tag:<22s} {c.dsl[:58]}")
            rows.append({"rank": i, "loss": c.loss, "channels": chs, "dsl": c.dsl})
    report["steps"]["channel_usage"]["top_largest_channel_group"] = rows

    # ---- 4c. 数值校准: 已知真值的合成算例 ----
    from lgf_agent.core import dsl as _dsl
    from lgf_agent.core.paramfit import fit_joint
    cases = _calibration_cases()
    vn = ["roloc", "T1"]
    g = _dsl.parse_dsl("C1 * sqrt(roloc) + C2 * T1", vn)
    fr = fit_joint(g, cases, n_starts=8, maxiter=200, seed=0)
    rep = analysis.leave_one_channel_out(g, fr.theta, cases, vn)
    ch = rep["channels"].get("T1", {})
    # 解析真值: contribution = 0.5*T1; pred_rms 由本算例数据直接给出
    pred = 2.0 * np.sqrt(np.concatenate([c["X"][:, 0] for c in cases])) \
        + 0.5 * np.concatenate([c["X"][:, 1] for c in cases])
    exact_share = float(np.sqrt(np.mean((0.5 * np.concatenate([c["X"][:, 1] for c in cases])) ** 2))
                        / np.sqrt(np.mean(pred ** 2)))
    print("\n  数值校准 (合成算例 y = 2√roloc + 0.5·T1, 已知真值):")
    print(f"    拟合 loss        = {rep['loss_full']:.3e}  (应 ≈ 0)")
    print(f"    T1 rms_share     = {ch.get('rms_share'):.5f}   "
          f"解析值 = {exact_share:.5f}")
    print(f"    去掉 T1 后 Δloss = {ch.get('delta_loss'):.5f}")
    print(f"    公式实际引用通道 = {rep['scalar_only'] and '无' or list(rep['channels'])}")
    report["steps"]["calibration"] = {"loss_full": rep["loss_full"],
                                      "t1_rms_share": ch.get("rms_share"),
                                      "exact_rms_share": exact_share,
                                      "delta_loss": ch.get("delta_loss"),
                                      "ranking": rep.get("ranking")}

    # 纯标量公式 -> 走 scalar_only 分支 (变量池里有 T1, 但公式没用)
    g2 = _dsl.parse_dsl("C1 * sqrt(roloc)", vn)
    fr2 = fit_joint(g2, cases, n_starts=8, maxiter=200, seed=0)
    rep2 = analysis.leave_one_channel_out(g2, fr2.theta, cases, vn)
    print(f"    纯标量式 (C1√roloc) → scalar_only = {rep2['scalar_only']} "
          f"(池里仍有 T1: unused_pool_channels = {rep2['unused_pool_channels']})")
    if abs((ch.get("rms_share") or 0.0) - exact_share) < 1e-6:
        print("    [校准通过] 数值分解与解析真值一致 (< 1e-6)")
    print("    [可复用] 对历史上那条 0.0418 的公式, 只要给它对应的 cases/var_names, "
          "调用 analysis.leave_one_channel_out 即可得到同类分解。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    help="缺省: 打包的 64 点/工况精简样例")
    ap.add_argument("--legacy", default=os.path.join(os.path.dirname(REPO),
                                                     "outputs", "asm"),
                    help="既有 runs_* 产物目录 (会递归找 *_final.json)")
    ap.add_argument("--out", default=default_out("p0_report"))
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

    # ---------- 步骤 4: 通道归因 ----------
    _step4_channel_attribution(archive, report)

    _dump(os.path.join(a.out, "p0_report.json"), report)
    print(f"\n[done] 报告 -> {os.path.join(a.out, 'p0_report.json')}")


if __name__ == "__main__":
    main()
