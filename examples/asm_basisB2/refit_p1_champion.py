"""P1/P2 冠军公式 → 终口径: 混合替换实验 (hybrid substitution).

背景: P1 三臂的 best_loss (0.0995 / 0.1076 / 0.2997) 是**中间口径** ——
单通道 a_j 的加权 NMSE。报告轮 (t23basisB2_paper) 证明终口径联合 refit
能把 10^0 级中间误差拉回 10^-1~10^-2 (抵消结构被治愈)。

本脚本做一件干净的事 (自包含, 只依赖本仓库与上层完整数据集):

    基线 B0: 4 通道全 lstsq (48 参数) 的重组 loss_vw —— stage_base 直接给出;
    上界 B2: 把 a_2 通道整个置零的重组 loss —— 该通道的重要性;
    实验 H1: a_2 换成 P1 冠军 SR 公式 (5-6 参数), 其余 3 通道保持 lstsq,
             对 SR 参数做联合 LM 抛光 (多起点), 终口径重组 NMSE。

判读: H1 越接近 B0 越好 —— "agent 发现的紧凑公式复现了 12 参数多项式
通道对 b23 的全部贡献"; 若 H1 < B0 则是严格正面结果 (公式还更简洁)。

用法:
    python examples/asm_basisB2/refit_p1_champion.py                  # P1 三臂冠军
    python examples/asm_basisB2/refit_p1_champion.py --p2-dir <dir>   # 追加 P2 各 seed
输出: run/basisb2_p1_refit/refit_p1_champion.json + 控制台表。
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
    BASE_VARS, coef_pool, eval_nmse, load_channels, stage_base)
from lgf_agent.domains.asm.refit import compile_expr              # noqa: E402

from scipy.optimize import least_squares                          # noqa: E402


# ---------------------------------------------------------------------------
# 冠军提取
# ---------------------------------------------------------------------------
def _normalize_dsl(dsl: str) -> str:
    """kernel 参数有 C*/K* 两族, compile_expr 只认 C* —— 把 K<n> 重命名为 C7<n>。

    (C7x 避开 C1..C6 的常规占用; 参数初值由 multi-start 提供, 不依赖原名。)
    """
    import re
    def sub(m):
        return f"C7{m.group(1)}"
    return re.sub(r"\bK(\d+)\b", sub, dsl)


def _extract_champions(p1_dir: str, p2_dir: str | None):
    """返回 [(label, dsl, theta)]。label 供报告用。"""
    out = []
    # P1: legacy + 两 agent 臂
    lf = os.path.join(p1_dir, "legacy", "legacy_final.json")
    if os.path.exists(lf):
        d = json.load(open(lf, encoding="utf-8"))
        out.append(("P1/legacy", d["best"]["dsl"],
                    d["best"].get("theta")))
    for arm in ("agent_no_insp", "agent_insp"):
        hits = (glob.glob(os.path.join(p1_dir, arm, "**", "result.json"),
                          recursive=True) +
                [os.path.join(p1_dir, arm, "result.json")])
        hits = [p for p in hits if os.path.exists(p)]
        if hits:
            r = json.load(open(hits[0], encoding="utf-8"))
            top = r.get("top") or []
            if top:
                c = top[0]
                out.append((f"P1/{arm}", c["dsl"], c.get("theta")))
    # P2: 每个 seed 的每个臂
    if p2_dir and os.path.isdir(p2_dir):
        for p in sorted(glob.glob(os.path.join(p2_dir, "seed*", "*",
                                               "result.json")) +
                        glob.glob(os.path.join(p2_dir, "seed*", "legacy",
                                               "legacy_final.json"))):
            rel = os.path.relpath(p, p2_dir).replace("\\", "/")
            seed = rel.split("/")[0]
            name = rel.split("/")[1]
            if p.endswith("legacy_final.json"):
                d = json.load(open(p, encoding="utf-8"))
                out.append((f"P2/{seed}/legacy", d["best"]["dsl"],
                            d["best"].get("theta")))
            else:
                r = json.load(open(p, encoding="utf-8"))
                top = r.get("top") or []
                if top:
                    out.append((f"P2/{seed}/{name}", top[0]["dsl"],
                                top[0].get("theta")))
    return out


# ---------------------------------------------------------------------------
# 混合 refit
# ---------------------------------------------------------------------------
def hybrid_refit(cases, base, j_star: int, dsl: str, theta0,
                 scale: float, *, n_starts: int = 5, max_nfev: int = 20000):
    """a_j* 换成 SR 公式, 其余通道保持 lstsq, 对 SR 参数做联合 LM。

    pred_i = base_pred_i - lstsq贡献_i + scale·f(x_i;θ)·Ũ_j*(x_i)
    loss   = mean_i( mean(res_i²)/var_c_i )   —— 与 eval_nmse/loss_vw 同口径
    """
    pool = coef_pool()
    B = [np.column_stack([fn(c["f"]) for _, fn in pool]) for c in cases]
    coef_flat = np.asarray(base["coef_flat"], dtype=float)
    cmat = np.asarray(base["coef"], dtype=float)              # [j, k]
    base_pred = [Z @ coef_flat for Z in base["design"]]
    lstsq_contrib = [B[i] @ cmat[j_star] * cases[i]["U"][:, j_star]
                     for i in range(len(cases))]
    Uj = [c["U"][:, j_star] for c in cases]
    var_c = [c["var_c"] for c in cases]

    fn, params, th0_def = compile_expr(_normalize_dsl(dsl))
    Xs = [np.column_stack([c["f"][k] for k in BASE_VARS]) for c in cases]
    y = [c["b23"] for c in cases]
    nv_cases = len(cases)
    ncol = len(BASE_VARS)

    def loss_of(theta):
        tot = 0.0
        for i in range(nv_cases):
            pred = base_pred[i] - lstsq_contrib[i] + \
                scale * fn(*[Xs[i][:, k] for k in range(ncol)],
                           *theta) * Uj[i]
            res = pred - y[i]
            tot += float(np.mean(res ** 2) / var_c[i])
        return tot / nv_cases

    def resid(theta):
        r = []
        for i in range(nv_cases):
            pred = base_pred[i] - lstsq_contrib[i] + \
                scale * fn(*[Xs[i][:, k] for k in range(ncol)],
                           *theta) * Uj[i]
            ri = (pred - y[i]) / np.sqrt(var_c[i])
            r.append(np.nan_to_num(ri, nan=1e6, posinf=1e6, neginf=1e6))
        return np.concatenate(r)

    # 多起点: 记录 theta (若长度吻合) / 全 1 / 抖动 —— 全零初值会除零
    ones = np.ones(max(len(params), 1))
    inits = [ones]
    if theta0 and len(theta0) >= len(params):
        t0 = np.asarray(theta0[:len(params)], dtype=float)
        if np.all(np.isfinite(t0)) and np.any(t0 != 0):
            inits.insert(0, t0)
    rng = np.random.default_rng(0)
    while len(inits) < n_starts:
        b = inits[len(inits) % len(inits)]
        inits.append(b * (1 + 0.3 * rng.standard_normal(len(b))))

    best = None
    errs = []
    for t0 in inits:
        try:
            sol = least_squares(resid, t0, method="trf", max_nfev=max_nfev)
            l = loss_of(sol.x)
            if np.isfinite(l) and (best is None or l < best[0]):
                best = (l, sol.x.tolist())
        except Exception as e:                                # noqa: BLE001
            errs.append(f"{type(e).__name__}: {e}")
    if best is None and errs:
        print(f"    [debug] 所有起点失败: {errs[0][:120]}")
    return best  # (loss, theta) or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--variant", default="e", choices=["v", "g", "e"])
    ap.add_argument("--target", type=int, default=2, help="被替换的通道 a_j (1 起)")
    ap.add_argument("--p1-dir", default=default_out("basisb2_p1_arms"))
    ap.add_argument("--p2-dir", default=None,
                    help="P2 多种子目录 (可选, 会扫描 seed*/*)")
    ap.add_argument("--runs-dir",
                    default=os.path.join(os.path.dirname(REPO), "outputs",
                                         "asm", "runs_basisB2"),
                    help="历史 runs_basisB2 目录 (同帧 swap 对照用)")
    ap.add_argument("--swap-group", default="e_lgf_mg_big",
                    help="同帧对照: 把该历史组的 a_j 公式放进同一替换框架")
    ap.add_argument("--out", default=default_out("basisb2_p1_refit"))
    a = ap.parse_args()
    j_star = a.target - 1

    # ---- 数据与基线 ----
    paths = resolve_datasets(a.data)
    cases = load_channels(a.variant, paths, cnorm="percase")
    base = stage_base(cases)                       # 默认池, use_whiten=False
    # scale 与 sr_cases_for_target 逐字对齐
    tr_all = list(range(len(cases)))
    pooled = np.concatenate([base["targets"][j_star][i] for i in tr_all])
    scale = float(np.sqrt(np.mean(pooled ** 2))) or 1.0
    loss_b0 = base["loss_vw"]                      # 48 参数全 lstsq 基线
    # 上界: a_2 置零
    coef_flat = np.asarray(base["coef_flat"], dtype=float)
    J = base["J"]
    K = base["K"]
    zero = coef_flat.copy()
    blk = np.arange(j_star, K * J, J)              # channel j_star 的列
    zero[blk] = 0.0
    preds0 = [Z @ zero for Z in base["design"]]
    loss_b2 = eval_nmse(preds0, cases)["loss_vw"]
    print(f"[hybrid] variant={a.variant} 通道 a_{j_star+1}  "
          f"B0(全lstsq 48参, in-sample 上界) = {loss_b0:.4f}   "
          f"B2(a_2 置零上界) = {loss_b2:.4f}   "
          f"参照: 历史全-SR 终口径 SOTA = 0.0576 (e_lgf_mg_big)")

    # ---- 各冠军 ----
    champs = _extract_champions(a.p1_dir, a.p2_dir)
    if not champs:
        print("[hybrid] 未找到任何冠军 (检查 --p1-dir/--p2-dir)")
        return
    rows = []
    for label, dsl, theta in champs:
        best = hybrid_refit(cases, base, j_star, dsl, theta, scale)
        if best is None:
            print(f"  {label:<26} LM 失败")
            continue
        l, th = best
        rows.append(dict(label=label, dsl=dsl, loss_refit=l,
                         n_params=len(th), theta=th))
        tag = ("✅ 优于历史 SOTA 0.0576" if l < 0.0576 else
               ("≈ 历史SOTA 同级" if l < 0.08 else "⚠️ 劣于历史 SOTA"))
        print(f"  {label:<26} H1(替换+refit) = {l:.4f}   "
              f"({len(th)} 参数)  {tag}")

    # ---- 同帧 swap 对照: 历史组的 a_j 放进同一替换框架 ----
    if a.swap_group and os.path.isdir(a.runs_dir):
        try:
            from lgf_agent.domains.asm.refit import load_basisB2_run
            run = load_basisB2_run(a.runs_dir, a.swap_group)
            if int(run["J"]) > j_star and run["variant"] == a.variant:
                t = run["targets"][j_star]
                best = hybrid_refit(cases, base, j_star, t["dsl"],
                                    t.get("theta"), scale)
                if best is not None:
                    l, th = best
                    rows.append(dict(label=f"swap/{a.swap_group}",
                                     dsl=t["dsl"], loss_refit=l,
                                     n_params=len(th), theta=th))
                    print(f"  {'swap/'+a.swap_group:<26} H1(替换+refit) = "
                          f"{l:.4f}   ({len(th)} 参数)  ← 同帧对照基准")
                else:
                    print(f"  swap/{a.swap_group:<21} LM 失败")
        except Exception as e:                                # noqa: BLE001
            print(f"  swap 对照不可用: {type(e).__name__}: {str(e)[:100]}")

    os.makedirs(a.out, exist_ok=True)
    out_p = os.path.join(a.out, "refit_p1_champion.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(dict(variant=a.variant, target=a.target,
                       B0_lstsq=loss_b0, B2_zeroed=loss_b2,
                       champions=rows), f, ensure_ascii=False, indent=2)
    print(f"\n[done] -> {out_p}")


if __name__ == "__main__":
    main()
