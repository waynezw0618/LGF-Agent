"""lgf_agent.analysis — 候选级只读分析查询 (通道归因 / 变量集画像).

为什么需要这一层 (方案 §1.2 的收口)
----------------------------------
本工程历史上有一类问题**只能靠事后手写一次性脚本**回答, 最典型的是 "T23 口径争议":

    最优公式到底有没有真的用张量基通道 T⁽ᵐ⁾₂₃? 用到的通道各自贡献多少?

当时为它写了 `_tmp_t23_contrib.py`, 里面硬编码了公式的两项、手工把 5 个 case 的数组
按 `Nv` 切开、再手算 `rms_B / rms_pred`。换一条公式就得重写一遍。

`store.Candidate.var_names` 落地后, 这类问题应该变成**一行查询**:

- **结构查询** (不需要数据, 回灌历史产物即可用)
  `tensor_channels_used(var_names)` → 这条公式用了哪些通道; `[]` 表示纯标量表。
  这一条就足以回答 "前排公式是否用了 T 通道"。

- **数值查询** (需要该口径对应的数据)
  `leave_one_channel_out(...)` → 把某通道的输入变量整体置零, 重算预测与损失:

      contribution_c(x) = f(x) − f(x | vars_c := 0)
      rms_share_c       = ‖contribution_c‖ / ‖f‖          (跨工况 pooled)
      delta_loss_c      = loss(f 去掉通道 c) − loss(f)      (loss 口径 = 逐工况 NMSE)

  这正是历史结论 "去掉 T1s 后 0.0418 → 0.0617" 的机器化版本。定义对**任意图结构**成立:
  不要求线性求和, 不要求预先知道项的分组 —— 只要求"这条公式是否用到通道 c"这件事
  能由变量名判定 (见 `channel_of_var`)。

命名兼容
--------
本工程出现过三套通道命名, 本模块全部识别:

    T1 / T1_23 / T1s / T1s_23        → "T1"
    T5s_v / T5_v / T5v / T5s_v_23    → "T5(v)"
    T10s_e                          → "T10(e)"

非通道变量 (`eta2` / `zeta2` / `s3` / `so2` / `s2o2` / `ret` / `roloc` / `yw` 等) 一律返回
`None`, 因此 "纯标量公式" 会被正确判成零通道。
"""
from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

# `T5s_v` / `T5_v` / `T5v` / `T10s_e_23` ... 统一成一个通道标签 `T5(v)`
_CHAN_RE = re.compile(r"^T(?P<idx>\d+)(?:s)?(?:[_-]?(?P<var>[vge]))?$")


def channel_of_var(name: str) -> Optional[str]:
    """把特征名映射到张量通道标签; 非通道变量返回 ``None``.

    识别规则 (先剥掉可能的 ``_23`` / ``23`` 分量后缀, 再匹配 ``T<idx>[s][_variant]``):

        >>> channel_of_var("T1_23")
        'T1'
        >>> channel_of_var("T5s_v")
        'T5(v)'
        >>> channel_of_var("T10s_e")
        'T10(e)'
        >>> channel_of_var("yw") is None
        True
    """
    if not isinstance(name, str):
        return None
    s = name.strip()
    # 剥分量后缀: `T1_23` / `T1s23` / `T1s.23`
    s = re.sub(r"[._]?23$", "", s)
    # 兼容 `mT1_23` 这类 m-pool 前缀
    if s.startswith("m") and len(s) > 1 and s[1] == "T":
        s = s[1:]
    m = _CHAN_RE.match(s)
    if not m:
        return None
    idx = m.group("idx")
    var = m.group("var")
    return f"T{idx}({var})" if var else f"T{idx}"


def is_channel_var(name: str) -> bool:
    return channel_of_var(name) is not None


# 便捷别名 (`leave_one_channel_out` 的 `channel_of=` 关键字即引用它)
channel_of = channel_of_var


def channel_groups(var_names: Sequence[str]) -> "OrderedDict[str, List[int]]":
    """``var_names`` → ``{通道标签: [列索引...]}`` (保持出现顺序, 只含通道变量)。"""
    groups: "OrderedDict[str, List[int]]" = OrderedDict()
    for i, n in enumerate(var_names):
        ch = channel_of_var(n)
        if ch is not None:
            groups.setdefault(ch, []).append(i)
    return groups


def tensor_channels_used(var_names: Sequence[str]) -> List[str]:
    """这条公式用到的全部张量通道标签; 空列表 = 纯标量公式 (结构判定, 不需要数据)。"""
    return list(channel_groups(var_names).keys())


def has_tensor_channel(var_names: Sequence[str]) -> bool:
    return bool(channel_groups(var_names))


def scalar_vars(var_names: Sequence[str]) -> List[str]:
    """非通道 (标量 / 壁距 / 局部扩展) 变量名。"""
    return [n for n in var_names if not is_channel_var(n)]


def vars_used_in_graph(graph, var_names: Sequence[str]) -> List[str]:
    """图里**实际引用**到的变量名 (按 ``var_names`` 顺序去重)。

    与 "变量池里有哪些通道" 区分开: 池里有而公式没用, 与公式里用了但贡献≈0,
    是两件不同的事 —— 前者是结构判定 (本函数), 后者才需要数值分解
    (`leave_one_channel_out`)。
    """
    from .core.graph import walk
    idx = set()
    for _path, n in walk(graph.root):
        if getattr(n, "op", None) == "var":
            try:
                idx.add(int(n.attrs["index"]))
            except Exception:                     # noqa: BLE001
                continue
    return [var_names[i] for i in sorted(idx) if 0 <= i < len(var_names)]


# ===========================================================================
# 数值通道归因 (需要数据)
# ===========================================================================
def _case_theta(theta: np.ndarray, cg, ci: int) -> np.ndarray:
    """取第 ci 个 case 的局部参数向量 [globals..., case_params_ci...]。"""
    th = np.asarray(theta, dtype=float)
    return np.concatenate([th[: cg.n_g], th[cg.n_g + ci * cg.n_c: cg.n_g + (ci + 1) * cg.n_c]])


def _predict_all(graph, theta: np.ndarray, cases) -> List[np.ndarray]:
    from .core.compiler import CompiledGraph
    cg = CompiledGraph(graph, backend="numpy")
    preds = []
    for ci, c in enumerate(cases):
        preds.append(np.asarray(cg.eval_case(_case_theta(theta, cg, ci), c["X"]), dtype=float))
    return preds


def _pooled_loss(preds: Sequence[np.ndarray], cases) -> float:
    """逐工况 NMSE 再平均 —— 与 compiler/paramfit 的 loss 口径逐字一致。"""
    ls = []
    for p, c in zip(preds, cases):
        y = np.asarray(c["y"], dtype=float)
        w = c.get("w")
        if w is None:
            var_y = float(np.var(y)) + 1e-12
            mse = float(np.mean((p - y) ** 2))
        else:
            w = np.asarray(w, dtype=float)
            sw = float(w.sum())
            mu = float(np.dot(w, y) / sw)
            var_y = float(np.dot(w, (y - mu) ** 2) / sw) + 1e-12
            mse = float(np.dot(w, (p - y) ** 2) / sw)
        ls.append(mse / var_y)
    return float(np.mean(ls))


def leave_one_channel_out(graph, theta, cases, var_names: Sequence[str],
                          *, channel_of: Callable[[str], Optional[str]] = channel_of_var,
                          ) -> Dict[str, Any]:
    """逐通道 "净贡献" 分解 (leave-one-channel-out).

    对**公式实际引用**的每个通道 ``c``:
      - ``contribution_c(x) = f(x) − f(x | vars_c := 0)``  —— 该通道对预测的实际影响
      - ``rms_share_c = ‖contribution_c‖ / ‖f‖``           —— 跨工况 pooled 的 RMS 占比
      - ``delta_loss_c = loss(去掉通道 c) − loss(全式)``     —— 去掉它损失的代价

    为什么要以 "公式实际引用" 而非 "变量池里有" 为准: 池里有 T 通道而公式写成
    `C1*sqrt(roloc)` 时, 该通道的贡献按定义就是 0, 报出来只会制造噪声。真正的
    争议是 "公式里出现了 T 通道但它到底起不起作用" —— 那才需要数值分解, 由
    ``rms_share_c`` 回答 (出现但无效 → 趋近 0)。

    返回 ``{"loss_full", "pred_rms", "channels": {...}, "n_channels", "scalar_only",
    "pool_channels", "unused_pool_channels"}``。公式不引用任何通道时 ``channels``
    为空且 ``scalar_only=True``, 调用方不必特判。
    """
    names = list(var_names)
    full_preds = _predict_all(graph, theta, cases)
    loss_full = _pooled_loss(full_preds, cases)
    # pooled 预测 RMS (跨工况): 分母, 避免 case 数 / 点数不等造成偏移
    all_pred = np.concatenate([np.ravel(p) for p in full_preds]) if full_preds else np.zeros(0)
    pred_rms = float(np.sqrt(np.mean(all_pred ** 2))) if all_pred.size else 0.0

    pool = tensor_channels_used(names)
    index_of = {n: i for i, n in enumerate(names)}
    used_vars = vars_used_in_graph(graph, names)
    groups: "OrderedDict[str, List[int]]" = OrderedDict()
    for v in used_vars:
        ch = channel_of(v)
        if ch is not None:
            groups.setdefault(ch, []).append(index_of[v])

    out: Dict[str, Any] = {
        "loss_full": float(loss_full),
        "pred_rms": pred_rms,
        "n_channels": len(groups),
        "scalar_only": len(groups) == 0,
        "pool_channels": pool,
        "unused_pool_channels": [c for c in pool if c not in groups],
        "vars_used": used_vars,
        "channels": {},
    }
    if not groups:
        return out

    for ch, idx in groups.items():
        mod = []
        for c in cases:
            X = np.array(c["X"], dtype=float, copy=True)
            X[:, idx] = 0.0                       # 该通道的输入整体置零
            d = dict(c)
            d["X"] = X
            mod.append(d)
        rec: Dict[str, Any] = {"vars": [names[i] for i in idx], "n_vars": len(idx)}
        try:
            preds_c = _predict_all(graph, theta, mod)
            if not all(np.all(np.isfinite(p)) for p in preds_c):
                raise FloatingPointError("非有限值")
            loss_c = _pooled_loss(preds_c, cases)
            # contribution = f(x) − f(x | vars_c := 0); 逐 case 对齐后 pooled
            contrib = np.concatenate([np.ravel(f - g) for f, g in zip(full_preds, preds_c)])
            rms_c = float(np.sqrt(np.mean(contrib ** 2)))
            rec.update({
                "available": True,
                "loss_without": float(loss_c),
                "delta_loss": float(loss_c - loss_full),
                "rms": rms_c,
                "rms_share": float(rms_c / pred_rms) if pred_rms > 0 else float("nan"),
                "max_abs": float(np.max(np.abs(contrib))) if contrib.size else 0.0,
            })
        except Exception as e:                    # noqa: BLE001
            # 置零导致除零 / 溢出 (如 1/T1 型结构) —— 如实标记, 不伪装成 "贡献 0"
            rec.update({"available": False, "note": f"无法置零评估: {e!r}"})
        out["channels"][ch] = rec

    ranked = sorted(
        [c for c in out["channels"].values() if c.get("available")],
        key=lambda r: -abs(r.get("rms_share") or 0.0))
    out["ranking"] = [{"channel": ch, "rms_share": r.get("rms_share"),
                       "delta_loss": r.get("delta_loss")}
                      for ch, r in sorted(out["channels"].items(),
                                          key=lambda kv: -abs(kv[1].get("rms_share") or 0.0))
                      if r.get("available")]
    return out


def candidate_channel_usage(dsl: str, var_names: Sequence[str], *,
                            parse: bool = True) -> Dict[str, Any]:
    """单条候选的通道画像: **公式引用** 与 **变量池提供** 两栏分开报。

    公式引用需要解析 DSL (按 `var_names` 索引); 解析失败 (历史产物命名不齐) 时
    降级为 "只有池信息", 并置 ``parse_ok=False`` —— 不伪装成 "公式没用通道"。
    """
    vn = list(var_names or [])
    pool = tensor_channels_used(vn)
    formula: List[str] = []
    parse_ok = False
    if parse and dsl:
        try:
            from .core import dsl as _dsl
            g = _dsl.parse_dsl(str(dsl), vn)
            formula = sorted({ch for ch in
                              (channel_of_var(v) for v in vars_used_in_graph(g, vn))
                              if ch})
            parse_ok = True
        except Exception:                          # noqa: BLE001
            parse_ok = False
    return {"pool_channels": pool, "formula_channels": formula,
            "parse_ok": parse_ok,
            # 判定以公式为准; 解析不了时退回池口径 (并在 parse_ok 里标明)
            "channels": formula if parse_ok else pool,
            "uses_tensor_channel": bool(formula) if parse_ok else bool(pool)}


def summarize_channel_usage(cands: Iterable[Any]) -> Dict[str, Any]:
    """对一组候选 (``Candidate`` 或含 ``var_names`` 的 dict) 做通道使用画像.

    直接回答那类"事后一次性脚本"问题:
      - 多少条候选**公式里真的用了**张量通道?
      - 用了哪些通道, 各自被几条公式引用?
      - 零通道 (纯标量) 候选中最好的是谁?

    ``n_scalar_only`` / ``channels`` 以**公式口径**为准 (这是争议的原点);
    ``n_pool_has_channel`` 另给池口径, 两者之差就是 "池里有但没用" 的条数。
    """
    usage: Dict[str, List[str]] = OrderedDict()
    pool_usage: Dict[str, List[str]] = OrderedDict()
    scalar_only: List[Any] = []
    n = n_pool = n_parse_fail = 0
    for c in cands:
        vn = getattr(c, "var_names", None) or (c.get("var_names") if isinstance(c, dict) else None) or []
        cid = getattr(c, "id", None) or (c.get("id") if isinstance(c, dict) else None) or "?"
        loss = getattr(c, "loss", None) if not isinstance(c, dict) else c.get("loss")
        dsl = getattr(c, "dsl", None) if not isinstance(c, dict) else c.get("dsl")
        u = candidate_channel_usage(str(dsl or ""), vn)
        n += 1
        if not u["parse_ok"]:
            n_parse_fail += 1
        for ch in u["formula_channels"]:
            usage.setdefault(ch, []).append(cid)
        for ch in u["pool_channels"]:
            pool_usage.setdefault(ch, []).append(cid)
        if u["pool_channels"]:
            n_pool += 1
        if not u["channels"]:
            scalar_only.append({"id": cid, "loss": loss, "dsl": dsl})
    scalar_only.sort(key=lambda r: (r["loss"] is None,
                                    r["loss"] if r["loss"] is not None else 0.0))
    return {
        "n_candidates": n,
        "n_with_tensor_channel": n - len(scalar_only),      # 公式口径
        "n_scalar_only": len(scalar_only),                  # 公式口径
        "frac_scalar_only": (len(scalar_only) / n) if n else 0.0,
        "n_pool_has_channel": n_pool,                       # 池口径
        "n_parse_failed": n_parse_fail,
        "channels": {ch: {"n_candidates": len(ids), "candidate_ids": ids[:20],
                          "n_pool_candidates": len(pool_usage.get(ch, []))}
                     for ch, ids in sorted(usage.items(),
                                           key=lambda kv: -len(kv[1]))},
        "best_scalar_only": scalar_only[:5],
    }
