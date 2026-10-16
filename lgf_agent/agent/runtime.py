"""lgf_agent.agent.runtime — 运行时: 动作解析 / observation 压缩 / 审计 / 成本会计.

对应方案 §3.6。要点:
- **tool-call 解析**: 复用 core.llm_ops 的 `_strip_reasoning` + `safe_json_loads`
  (已适配 GLM-5.3 thinking、```json 围栏、尾随逗号), 再加"动作序列"schema 校验
  与一次修复重试。
- **observation 压缩**: 全量 evidence 落 archive, prompt 里只放 ≤400 字符紧凑视图。
- **provenance & resume**: 每次工具调用 → traj.jsonl; 中断后从 archive + 最后一步恢复。
- **成本会计**: tokens / usd / calls 逐调用记账 (legacy 完全没有)。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..core.llm_ops import _strip_reasoning, safe_json_loads

# 动作序列 schema
ACTION_REQUIRED = ("tool",)

# 哨兵: 区分"解析失败"与"合法空对象" (safe_json_loads 把 default=None 视为 {})
_MISS = object()


class ParseError(ValueError):
    pass


def parse_agent_response(text: str, *, tool_names: Sequence[str] = ()
                         ) -> Dict[str, Any]:
    """把 LLM 文本响应解析为 {thought, actions, candidates, notes}。

    鲁棒性 (对齐 legacy 的 `safe_json_loads` 策略): 先剥 reasoning, 再去 ```json
    围栏, 再取首个平衡花括号块; 解析失败返回空骨架并置 `parse_error`。
    """
    raw = _strip_reasoning(text or "")
    obj = safe_json_loads(raw, _MISS)
    if obj is _MISS or not isinstance(obj, dict):
        return {"thought": "", "actions": [], "candidates": [],
                "parse_error": "no_json_object", "raw": raw[:600]}
    actions = obj.get("actions") or []
    cands = obj.get("candidates") or obj.get("graphs") or []
    norm_actions = []
    for a in actions:
        if isinstance(a, str):
            norm_actions.append({"tool": a, "args": {}})
            continue
        if not isinstance(a, dict):
            continue
        tool = a.get("tool") or a.get("name")
        if not isinstance(tool, str):
            continue
        args = a.get("args")
        if args is None:
            args = {k: v for k, v in a.items() if k not in ("tool", "name", "args")}
        norm_actions.append({"tool": tool, "args": args if isinstance(args, dict) else {}})
    if tool_names:
        unknown = [a["tool"] for a in norm_actions if a["tool"] not in tool_names]
        if unknown:
            obj["unknown_tools"] = unknown
    norm_c = []
    for c in cands:
        if isinstance(c, str):
            norm_c.append({"dsl": c})
        elif isinstance(c, dict) and isinstance(c.get("dsl"), str):
            norm_c.append(c)
    out = {"thought": str(obj.get("thought") or obj.get("reasoning") or ""),
           "actions": norm_actions, "candidates": norm_c,
           "notes": obj.get("notes") or ""}
    if "unknown_tools" in obj:
        out["unknown_tools"] = obj["unknown_tools"]
    return out


# ---------------------------------------------------------------------------
class Audit:
    """append-only 轨迹审计 (traj.jsonl) —— 也是 resume 的依据。"""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.records: List[Dict[str, Any]] = []

    def log(self, **rec: Any) -> Dict[str, Any]:
        rec.setdefault("ts", time.time())
        self.records.append(rec)
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return rec

    @classmethod
    def replay(cls, path: str) -> "Audit":
        a = cls(path)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            a.records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return a

    def resume_point(self) -> Dict[str, Any]:
        """返回最后一次 (r, c, ell) 与已提交候选 id 列表 (供断点续跑)."""
        last = {}
        submitted: List[str] = []
        for rec in self.records:
            if rec.get("event") == "step":
                last = {k: rec.get(k) for k in ("r", "c", "ell")}
            if rec.get("event") == "submit":
                cid = (rec.get("candidate") or {}).get("id")
                if cid:
                    submitted.append(cid)
        return {"last_step": last, "submitted": submitted}


# ---------------------------------------------------------------------------
@dataclass
class CostLedger:
    tokens: int = 0
    usd: float = 0.0
    calls: int = 0
    seconds: float = 0.0
    per_role: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def add(self, role: str, *, tokens: int = 0, usd: float = 0.0,
            seconds: float = 0.0) -> None:
        self.tokens += int(tokens)
        self.usd += float(usd)
        self.seconds += float(seconds)
        self.calls += 1
        r = self.per_role.setdefault(role, {"tokens": 0, "usd": 0.0,
                                            "calls": 0, "seconds": 0.0})
        r["tokens"] += int(tokens)
        r["usd"] += float(usd)
        r["calls"] += 1
        r["seconds"] += float(seconds)

    def to_dict(self) -> Dict[str, Any]:
        return dict(tokens=self.tokens, usd=round(self.usd, 6), calls=self.calls,
                    seconds=round(self.seconds, 3), per_role=self.per_role)


def estimate_tokens(text: str) -> int:
    """粗估 token 数 (中英混排 ~2.2 字符/token); 真实用量以后端返回为准。"""
    return max(1, int(len(text or "") / 2.2))


def clip(text: str, n: int = 400) -> str:
    s = " ".join(str(text).split())
    return s if len(s) <= n else s[: n - 1] + "…"
