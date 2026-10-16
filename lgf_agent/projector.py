"""lgf_agent.projector — L2 投影视图 P_ρ(S_t).

SRHarness 消融的硬数字 (方案 §1.2): 把 Pareto 视图换成"单一最优公式视图",
SA 从 **74.47% → 66.67%**; 而**移除基于表达式的视图**掉到 67.57% (掉幅最大的一档)。

我们的 `M_GUIDE` 恰好只喂 `best_g` 单点 —— 正是文献证明会让精炼**偏向早期解**
的 single-best 配置。所以:
- `pareto`      → 主视图 (默认)
- `topk`        → 次要信号
- `single_best` → **只用于复现文献消融 / 作为 M_GUIDE 现状对照**

投影同时承担"有界化": 每次注入 prompt 的候选数上限 `max_n`。
被省略的候选仍在 archive 中, 后续可重新纳入。
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Sequence

from .store import Candidate


# ---------------------------------------------------------------------------
# 非支配前沿
# ---------------------------------------------------------------------------
def pareto_front(cands: Sequence[Candidate]) -> List[Candidate]:
    """(loss ↓, complexity ↓) 二维非支配前沿 (O(n²), 候选规模 ≤ 10³ 足够).

    同 dsl 的重复候选 (多次跑批的同一公式) 先去重, 只保留 loss 最小的那条 ——
    否则回灌历史产物后前沿会被重跑副本淹没。
    """
    dedup: Dict[str, Candidate] = {}
    for c in cands:
        if c.loss >= float("inf"):
            continue
        prev = dedup.get(c.dsl)
        if prev is None or c.loss < prev.loss:
            dedup[c.dsl] = c
    pts = list(dedup.values())
    front: List[Candidate] = []
    for c in pts:
        dominated = False
        for o in pts:
            if o is c:
                continue
            if (o.loss <= c.loss and o.n_nodes <= c.n_nodes
                    and (o.loss < c.loss or o.n_nodes < c.n_nodes)):
                dominated = True
                break
        if not dominated:
            front.append(c)
    return sorted(front, key=lambda c: (c.loss, c.n_nodes))


def _proj_pareto(cands, max_n, rng):
    return pareto_front(cands)[:max_n]


def _proj_topk(cands, max_n, rng):
    return sorted(cands, key=lambda c: c.loss)[:max_n]


def _proj_single_best(cands, max_n, rng):
    return sorted(cands, key=lambda c: c.loss)[:1]


PROJECTORS: Dict[str, Callable[[Sequence[Candidate], int, random.Random],
                               List[Candidate]]] = {
    "pareto": _proj_pareto,
    "topk": _proj_topk,
    "single_best": _proj_single_best,
}


def project(cands: Sequence[Candidate], *, view: str = "pareto",
            max_n: int = 8, seed: int = 0,
            shuffle: bool = True, admissible_only: bool = False,
            rng: Optional[random.Random] = None) -> List[Candidate]:
    """把候选集合投影为"注入 prompt 的有界视图"。

    Parameters
    ----------
    view : {"pareto", "topk", "single_best"}
    max_n : 上界 (有界化)
    shuffle : 呈现顺序随机化 —— 缓解 LLM 位置偏好 (方案 §5.2 风险 R1)。
              仅打乱顺序, 不改变入选集合。
    """
    if view not in PROJECTORS:
        raise ValueError(f"未知投影视图: {view!r} (可选 {sorted(PROJECTORS)})")
    pool = [c for c in cands
            if c.loss is not None and c.loss < float("inf")
            and (not admissible_only or c.is_admissible())]
    r = rng or random.Random(seed)
    out = list(PROJECTORS[view](pool, max(1, int(max_n)), r))
    if shuffle:
        r.shuffle(out)
    return out


def render(cands: Sequence[Candidate], *, view: str = "pareto",
           max_n: int = 8, header: bool = True) -> str:
    """把投影结果渲染为 prompt 文本块 (紧凑视图)."""
    picked = project(cands, view=view, max_n=max_n, shuffle=False)
    if not picked:
        return "(archive 为空)"
    lines: List[str] = []
    if header:
        lines.append(f"# 候选池视图 [{view}] · 共 {len(cands)} 条, 展示 {len(picked)} 条")
    lines.extend(c.compact() for c in picked)
    return "\n".join(lines)
