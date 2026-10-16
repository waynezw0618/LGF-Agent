"""lgf_agent.domains.asm.dataset — ASM 数据加载 (dataset_trueasm_*.npz / dataset_*.npz).

支持两种既有布局
----------------
1. **trueasm** (`dataset_trueasm_{v,g,e}.npz`): 全量 T1..T10 + 5 个 Pope 不变量
   + Re_t/Ro_loc。键: `c{i}_ro / c{i}_yh / c{i}_valid / c{i}_b6 /
   c{i}_scalars(,7) / c{i}_T1..T10 (N,6)`。
2. **pope/通用** (`dataset*.npz`): `c{i}_b6 / c{i}_valid / c{i}_featB` +
   `featB_keys` + `b23` 通过 `c{i}_a` 或 `c{i}_b6[:,5]`。

统一的输出是"每个工况一个 case", 其中:
    X = Θ (Pope 不变量 + Re_t + Ro_loc [+ yw])
    y = b23 (b 的 23 分量)
    w = 由 `weighting` 决定 (默认 None → 逐工况 NMSE, 与 loss_vw 口径逐字一致)

关于 `w` 的两种口径 (重要, 别混用)
-----------------------------------
`weighting="nmse"` (默认): **不设 w**。此时 `compiler.eval_cases` 走逐工况分支
    loss = mean_c[ mean_y(res²) / var_c(b23) ]
这正是本工程的统一评估口径 `loss_vw` (见 rc_asm_vw_only.eval_vw_predictions),
也是 V4b 冠军族 (0.1154) 所用的口径。

`weighting="channel"`: w_i = T̃23,i² / (Nv·var_c)。这使"加权系数误差"逐点等于该
通道对 b23 的 NMSE 贡献, 但会引入 **T̃23² 的动态范围** (本数据集实测 1.8e8 倍),
使参数拟合病态。**只在与白化通道 (basisB2) 配套时使用**, 不要直接套在未归一的 b23 上。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .channels import COMP6, INDEX23, NONZERO_CHANNELS_23
from .invariants import SCALAR_COLUMNS, variables_for

RO_CASES: List[float] = [7.5, 15.0, 30.0, 75.0, 150.0]


@dataclass
class CaseExtra:
    """约束探针需要的、不进 X/y 的附加量 (随 case 一起传给 VerifyContext.extras)."""
    ro: float
    yh: np.ndarray
    b23: np.ndarray
    T23: np.ndarray            # 原始 T̃23 (未归一)
    T23n: np.ndarray           # 归一后 (跨工况 pooled RMS)
    var_c: float
    channels: Optional[np.ndarray] = None    # 多通道: (N, Jch)
    channel_names: List[str] = field(default_factory=list)


def _find_trueasm(root: str, variant: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    cand = os.path.join(root, f"dataset_trueasm_{variant}.npz")
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(f"找不到 {cand}; 请用 dataset= 指定 npz 路径")


def load_npz(path: str) -> "np.lib.npyio.NpzFile":
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据集不存在: {path}")
    return np.load(path, allow_pickle=True)


def _per_case_keys(ds, i: int) -> str:
    return f"c{i}_"


def load_cases(path: str, *, variant: str = "g",
               channel_mode: str = "single",
               var_mode: str = "minimal",
               channel_norm: str = "pooled",
               ro_cases: Sequence[float] = tuple(RO_CASES),
               with_wall: bool = True,
               weighting: str = "nmse",
               decoy_components: Sequence[int] = ()) -> Dict[str, Any]:
    """加载并构造 cases。

    channel_mode:
        "single" → 目标 = b23, 权重用单个 T̃23 (即 b23 = g(Θ)·T̃23 口径)
        "multi"  → 同时在 extras 里附带四通道 (供 C2 可表性探针 / 多通道变体)
    var_mode: 见 invariants.variables_for ("minimal" | "full" | "pope_only")
    channel_norm: "pooled"(跨工况统一 RMS) | "percase"(逐工况 RMS)
    with_wall: 是否把 yw 计入自变量 (var_mode="minimal" 时默认含 yw)
    weighting: "nmse"(默认, 无权重, 逐工况 NMSE) | "channel"(T̃23² 加权, 病态, 慎用)
    """
    ds = load_npz(path)
    has_scalars = f"c0_scalars" in ds.files
    var_names = variables_for(var_mode)
    if not with_wall:
        var_names = [v for v in var_names if v != "yw"]

    n_cases = len(ro_cases)
    raw: List[Dict[str, Any]] = []
    for i in range(n_cases):
        p = _per_case_keys(ds, i)
        valid = ds[p + "valid"].astype(bool)
        yh = ds[p + "yh"][valid].astype(float)
        ro = float(np.asarray(ds[p + "ro"]).ravel()[0]) if (p + "ro") in ds.files else ro_cases[i]

        if (p + "b6") in ds.files:
            b23 = ds[p + "b6"][valid, INDEX23].astype(float)
        else:
            b23 = ds[p + "a"][valid, INDEX23].astype(float)

        if has_scalars:
            sc = ds[p + "scalars"][valid].astype(float)
            col = {name: sc[:, k] for k, name in enumerate(SCALAR_COLUMNS)}
        else:
            # 通用布局: 从 featB 中按名字取 (best-effort)
            fk = [str(x) for x in ds["featB_keys"]] if "featB_keys" in ds.files else []
            fb = ds[p + "featB"][valid].astype(float) if (p + "featB") in ds.files else None
            col = {}
            for name in SCALAR_COLUMNS:
                col[name] = fb[:, fk.index(name)] if (fb is not None and name in fk) \
                    else np.full(int(valid.sum()), np.nan)

        col["yw"] = 1.0 - np.abs(yh)

        chans, names = [], []
        for m in NONZERO_CHANNELS_23:
            key = p + f"T{m}"
            if key not in ds.files:
                continue
            arr = np.asarray(ds[key])
            # 兼容两种既有布局: 全量 npz 给 (N,6); 精简样例只存 23 分量 (N,)
            colvec = arr[valid, INDEX23] if arr.ndim == 2 else arr[valid]
            chans.append(colvec.astype(float))
            names.append(f"T{m}")
        channels = np.column_stack(chans) if chans else None

        # 诱饵目标 (反作弊 C9): 同 X 的其它分量
        decoys_i: Dict[int, np.ndarray] = {}
        for comp in decoy_components:
            if (p + "b6") in ds.files:
                decoys_i[int(comp)] = ds[p + "b6"][valid, int(comp)].astype(float)

        raw.append(dict(ro=ro, yh=yh, b23=b23, col=col, channels=channels,
                        names=names, var_c=float(np.var(b23)),
                        Nv=int(np.sum(valid)), decoys=decoys_i))

    # 通道归一 (跨工况 pooled RMS 或逐工况)
    if raw and raw[0]["channels"] is not None:
        if channel_norm == "percase":
            for c in raw:
                rr = np.sqrt(np.mean(c["channels"] ** 2, axis=0))
                rr[rr <= 0] = 1.0
                c["_rms"] = rr
                c["_ch_n"] = c["channels"] / rr
        else:
            pooled = np.vstack([c["channels"] for c in raw])
            rr = np.sqrt(np.mean(pooled ** 2, axis=0))
            rr[rr <= 0] = 1.0
            for c in raw:
                c["_rms"] = rr
                c["_ch_n"] = c["channels"] / rr

    cases: List[dict] = []
    extras: List[CaseExtra] = []
    decoys: Dict[str, List[dict]] = {COMP6[int(k)]: [] for k in decoy_components}
    for c in raw:
        X = np.column_stack([c["col"][v] for v in var_names])
        y = c["b23"]
        if c.get("_ch_n") is not None:
            T23n = c["_ch_n"][:, 0]            # 第一通道 = T1 (与既有 CHANS 首元素一致)
        else:
            T23n = np.ones_like(y)
        nv = max(1, c["Nv"])
        var_c = max(c["var_c"], 1e-300)
        if weighting == "channel":
            w = (T23n ** 2) / (nv * var_c)
        else:
            w = None                            # 逐工况 NMSE (loss_vw 口径)
        base = dict(X=X, y=y, name=f"Ro{c['ro']:g}")
        if w is not None:
            base["w"] = w
        cases.append(base)
        extras.append(CaseExtra(ro=c["ro"], yh=c["yh"], b23=y, T23=T23n,
                                T23n=T23n, var_c=c["var_c"],
                                channels=c.get("_ch_n"),
                                channel_names=list(c["names"])))
        for comp_idx, arr in (c.get("decoys") or {}).items():
            name = COMP6[int(comp_idx)]
            dv = max(float(np.var(arr)), 1e-300)
            db = dict(X=X, y=arr, name=f"Ro{c['ro']:g}")
            if weighting == "channel":
                db["w"] = (T23n ** 2) / (nv * dv)
            decoys[name].append(db)

    return {"cases": cases, "var_names": var_names, "extras": extras,
            "decoys": decoys, "path": path, "variant": variant,
            "channel_mode": channel_mode, "channel_norm": channel_norm,
            "weighting": weighting}


def evaluate_b23_nmse(preds: Sequence[np.ndarray],
                      extras: Sequence[CaseExtra]) -> Dict[str, Any]:
    """统一评估口径: loss_vw = mean_c[ mean_y (b23_pred - b23)² / var_c(b23) ]."""
    per, nms = {}, []
    for p, e in zip(preds, extras):
        r = float(np.mean((np.asarray(p) - e.b23) ** 2) / (e.var_c + 1e-300))
        nms.append(r)
        per[f"Ro{e.ro:g}"] = round(r, 4)
    return {"loss_vw": float(np.mean(nms)), "worst": float(np.max(nms)),
            "per_case": per}
