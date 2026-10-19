"""lgf_agent.domains.asm.refit — 终口径: 固定形状的联合常数重拟合.

报告 (BasisB2_full_matrix_report.md §1) 的两级口径:
- **中间口径**: SR 逐通道独立估常数后直接重组 — 方法对比用;
- **终口径 (joint refit)**: 固定 SR 公式**形状**, 把 J 条公式的全部常数联合
  Levenberg–Marquardt 最小化重组 NMSE。PySR 的内联数值常量同样参数化后参与
  同一联合优化 — **终口径对两方法完全公平** (报告 §1 两级口径定义)。

为什么需要这一步 (报告 §4.3, 误差机理):
中间口径 10⁰ 级误差的 84–92% 来自"逐通道独立估常数 × 抵消结构"
(抵消比 Σ‖C_m‖/‖b23‖ = 99–32201×)。联合重拟合后全部回落 1–2 个量级 —
**SR 找形状 + 联合定常数**是系数口径的正确两阶段用法。

本模块与上层 scripts/rc_asm_basisB2.py 的 joint_refit/_compile_expr 数值对齐,
但不依赖上层工程 — 可独立打包发布。验证锚点: e_lgf_mg_big refit = 0.0576
(worst 0.068), g_lgf_mg_big = 0.1238, m_lgf_go_big = 0.0722。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .basisb2 import (BASE_VARS, load_channels, RO_CASES, eval_nmse,
                      whiten)

__all__ = ["compile_expr", "joint_refit", "load_basisB2_run",
           "replay_run_refit", "collect_matrix"]


# ---------------------------------------------------------------------------
# 公式编译: sympy 表达式 → 可调用 fn(X_cols..., params...)
# ---------------------------------------------------------------------------
def compile_expr(dsl: str):
    """把 DSL / sympy 公式字符串编译为 fn(*X_cols, *params)。

    - **LGF 公式**: 参数是显式 `C<k>` 符号 (theta 由调用方给);
    - **PySR 公式**: 数值常量内联 → 逐个 Float atom 替换为符号 `C9xx`
      (初值 = 原数值), 使 PySR 与 LGF 在联合重拟合中同场抛光。

    返回 (fn, param_names, theta0)。
    """
    import sympy as sp

    s = str(dsl).replace("^", "**")
    sy = {n: sp.Symbol(n) for n in BASE_VARS}
    expr = sp.sympify(s, locals=sy)
    repl, theta0, names = [], [], []
    for k, v in enumerate(sorted(expr.atoms(sp.Float), key=str)):
        nm = f"C9{90 + k}"
        repl.append((v, sp.Symbol(nm)))
        names.append(nm)
        theta0.append(float(v))
    if repl:
        expr = expr.xreplace(dict(repl))
    free = sorted({str(v) for v in expr.free_symbols if str(v).startswith("C")},
                  key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)
    syms = [sy[n] for n in BASE_VARS] + [sp.Symbol(c) for c in free]
    fn = sp.lambdify(syms, expr, printer=None,
                     modules=[{"inv": lambda x: 1.0 / x}, "numpy"])
    th0 = [dict(zip(names, theta0)).get(c, 0.0) for c in free]
    return fn, free, th0


def _eval_a(fn, X: np.ndarray, th: Sequence[float]) -> np.ndarray:
    """安全求值: NaN/inf → 0/±1e6 截断 (与上层 _eval_a 一致)。"""
    with np.errstate(all="ignore"):
        v = fn(*[X[:, i] for i in range(X.shape[1])], *th)
    v = np.asarray(v, dtype=float)
    if v.shape != (X.shape[0],):                     # 常数表达式广播
        v = np.broadcast_to(v, (X.shape[0],)).copy()
    return np.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)


# ---------------------------------------------------------------------------
# 联合重拟合
# ---------------------------------------------------------------------------
def joint_refit(cases: Sequence[Dict[str, Any]],
                dsls: Sequence[str],
                scales: Sequence[float],
                thetas: Optional[Sequence[Optional[Sequence[float]]]] = None,
                *, max_nfev: int = 20000,
                verbose: bool = False) -> Optional[Dict[str, Any]]:
    """固定 SR 公式形状, J 条公式的全部参数联合 LM 最小化重组 NMSE。

    Parameters
    ----------
    cases : basisb2.load_channels 的输出 (每 case 含 f/b23/U/Nv/var_c)。
    dsls : J 条公式 (LGF DSL 或 PySR equation 字符串)。
    scales : J 条通道的 s_j (β 靶标 RMS)。
    thetas : 各公式初值 (LGF 的 result theta); None 条目用编译初值。

    Returns
    -------
    dict(loss_vw/worst/per_case/theta/n_params/...) 或 None (全部失败)。
    """
    try:
        from scipy.optimize import least_squares
    except ImportError:
        raise SystemExit("joint_refit 需要 scipy")

    J = len(dsls)
    fns, prs, th0 = [], [], []
    for j in range(J):
        fn, params, th0_p = compile_expr(dsls[j])
        th = thetas[j] if thetas is not None else None
        if th and len(th) >= len(params):
            th0_p = [float(x) for x in th[:len(params)]]
        fns.append(fn)
        prs.append(params)
        th0.extend(th0_p)
    seg, o = [], 0
    for j in range(J):
        seg.append((o, o + len(prs[j])))
        o += len(prs[j])
    if o == 0:
        return None
    th0 = np.asarray(th0, dtype=float)

    def recon(th, c) -> np.ndarray:
        X = np.column_stack([c["f"][k] for k in BASE_VARS])
        acc = np.zeros(c["Nv"])
        for j in range(J):
            a = _eval_a(fns[j], X, th[seg[j][0]:seg[j][1]])
            acc = acc + a * scales[j] * c["U"][:, j]
        return acc

    def resid(th):
        return np.concatenate([
            (recon(th, c) - c["b23"]) / np.sqrt(c["var_c"]) for c in cases])

    try:
        sol = least_squares(resid, th0, method="lm", max_nfev=max_nfev)
    except Exception as e:                                    # noqa: BLE001
        if verbose:
            print(f"[refit] LM 失败: {type(e).__name__}: {e}")
        return None
    th1 = np.clip(sol.x, -1e8, 1e8)

    preds = [recon(th1, c) for c in cases]
    r = eval_nmse(preds, cases)
    out = dict(loss_vw=r["loss_vw"], worst=r["worst"], per_case=r["per_case"],
               theta=[float(v) for v in th1], n_params=int(o),
               dsls=list(dsls), scales=[float(s) for s in scales])
    if verbose:
        print(f"[refit] loss={r['loss_vw']:.4f} worst={r['worst']:.4f} "
              f"(n_par={o}) | " +
              " ".join(f"{k[2:]}:{v:.3f}" for k, v in r["per_case"].items()))
    return out


def refit_intermediate(cases, dsls, scales, thetas=None) -> Dict[str, Any]:
    """中间口径评估: 用 SR 自带常数 (不联合抛光) 直接重组。"""
    J = len(dsls)
    fns, segs, th = [], [], []
    o = 0
    for j in range(J):
        fn, params, th0_p = compile_expr(dsls[j])
        t = thetas[j] if thetas is not None else None
        if t and len(t) >= len(params):
            th0_p = [float(x) for x in t[:len(params)]]
        fns.append(fn)
        segs.append((o, o + len(params)))
        th.extend(th0_p)
        o += len(params)
    th = np.asarray(th, dtype=float)
    preds = []
    for c in cases:
        X = np.column_stack([c["f"][k] for k in BASE_VARS])
        acc = np.zeros(c["Nv"])
        for j in range(J):
            a = _eval_a(fns[j], X, th[segs[j][0]:segs[j][1]])
            acc = acc + a * scales[j] * c["U"][:, j]
        preds.append(acc)
    return eval_nmse(preds, cases)


# ---------------------------------------------------------------------------
# 上层工程 result.json 读取 (回放历史冠军)
# ---------------------------------------------------------------------------
def load_basisB2_run(runs_dir: str, group: str) -> Dict[str, Any]:
    """读 outputs/asm/runs_basisB2/<group>/result.json (上层工程产物)。

    返回 dict: variant/method/targets(dsl/scale/theta)/refit/J/chan_names/...
    """
    p = os.path.join(runs_dir, group, "result.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"找不到 {p}")
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def replay_run_refit(runs_dir: str, group: str, dataset_paths: Dict[str, str],
                     *, cnorm: str = "percase",
                     use_run_refit_as_init: bool = True,
                     verbose: bool = True) -> Dict[str, Any]:
    """回放一个历史组: 冠军 DSL → 重做联合 refit → 与记录值对账。

    这是 example `run_refit_replay.py` 的核心: 证明 LGF-Agent 包能**独立复算**
    上层工程的终口径数字 (0.0576 / 0.1238 / 0.0722), 而不是引用它。
    """
    run = load_basisB2_run(runs_dir, group)
    variant = run["variant"]
    J = int(run["J"])
    rows = run["targets"]
    dsls = [r["dsl"] for r in rows]
    scales = [float(r["scale"]) for r in rows]
    thetas = [r.get("theta") for r in rows]
    if use_run_refit_as_init:
        rf = run.get("refit") or {}
        if rf.get("theta") and len(rf["theta"]) == sum(
                len(compile_expr(d)[1]) for d in dsls):
            thetas = _split_theta(rf["theta"], dsls)
    cases = load_channels(variant, dataset_paths, cnorm=cnorm)
    whiten(cases, use_whiten=False)          # 写入 U 键 (= RMS 归一通道)
    # 中间口径用 SR 原始 theta; 终口径优先用 run 记录的 refit theta 作初值
    inter = refit_intermediate(cases, dsls, scales,
                               [r.get("theta") for r in rows])
    refit = joint_refit(cases, dsls, scales, thetas, verbose=verbose)
    return {"group": group, "variant": variant, "J": J,
            "chan_names": run.get("chan_names"),
            "recorded": {"intermediate": run.get("loss_vw"),
                         "refit": (run.get("refit") or {}).get("loss_vw")},
            "replayed": {"intermediate": inter["loss_vw"],
                         "refit": (refit or {}).get("loss_vw")},
            "refit_detail": refit, "intermediate_detail": inter,
            "run": run}


def _split_theta(flat: Sequence[float], dsls: Sequence[str]):
    """把 refit.theta (扁平) 按各公式的参数数切开。"""
    out, o = [], 0
    for d in dsls:
        _, params, _ = compile_expr(d)
        out.append([float(v) for v in flat[o:o + len(params)]])
        o += len(params)
    return out


def collect_matrix(runs_dir: str) -> List[Dict[str, Any]]:
    """汇总 runs_basisB2/*/result.json → 报告 §3.1 的 27 组矩阵 (纯读取)。"""
    import glob as _glob
    rows = []
    for p in sorted(_glob.glob(os.path.join(runs_dir, "*", "result.json"))):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:                                      # noqa: BLE001
            continue
        rf = d.get("refit") or {}
        rows.append(dict(
            group=os.path.basename(os.path.dirname(p)),
            variant=d.get("variant"), method=d.get("method"),
            backend=d.get("backend"), tag=(d.get("method") or "") ,
            J=d.get("J"), n_targets=len(d.get("targets") or []),
            loss_mid=d.get("loss_vw"), worst_mid=d.get("worst"),
            loss_refit=rf.get("loss_vw"), worst_refit=rf.get("worst"),
            per_case_refit=rf.get("per_case"),
            n_params_refit=rf.get("n_params"),
            prompt=d.get("prompt"), pop=d.get("pop"), gens=d.get("gens"),
            maxparams=d.get("maxparams")))
    return rows
