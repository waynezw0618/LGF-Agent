"""lgf_agent.store — L2 状态层: 候选-证据契约 + append-only archive.

为什么需要这一层 (方案 §1.2)
---------------------------
现状是候选只活在内存 `pop` 里, 跨进程即失, 落盘只有 top-8。后果已经显现过:
- T23 口径争议: 判断"最优公式是否真用了 T23"只能事后手写 `_tmp_t23_contrib.py`;
- V4b 的 "M_GUIDE alive=0": LLM 干预到底有没有生效**不可观测**;
- 跨 run 复用为零: 同族候选无法做统一 Pareto 排序。

本层提供两件东西:
1. `Candidate` —— 候选-证据契约 (公式 + 参数 + 指标 + 约束回执 + 归因 + 溯源);
   evidence 全量落盘, 而注入 prompt 时只取压缩视图 (见 projector)。
2. `Archive` —— append-only JSONL, 可审计/可 diff/天然断点续跑
   (`runner_sm_resume.ps1` 的直接需求)。
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

# 候选状态机: proposed → fitted → admitted | rejected ; admitted → submitted
STATUSES = ("proposed", "fitted", "admitted", "rejected", "submitted")


_ID_SEQ = {"n": 0}


def _new_id(prefix: str = "c") -> str:
    """进程内唯一 id: 递增序号 + 随机后缀 (避免大批量回灌时的生日碰撞)。"""
    _ID_SEQ["n"] += 1
    return f"{prefix}{_ID_SEQ['n']:05d}{uuid.uuid4().hex[:4]}"


@dataclass
class Candidate:
    """候选-证据契约 (方案 §3.2)."""

    id: str = field(default_factory=_new_id)
    dsl: str = ""
    var_names: List[str] = field(default_factory=list)

    # 拟合结果
    theta: List[float] = field(default_factory=list)
    param_names: List[str] = field(default_factory=list)
    loss: float = float("inf")

    # 度量 (loss_vw / r2 / rho / nmse_per_case ...)
    metrics: Dict[str, Any] = field(default_factory=dict)
    # 复杂度 (nodes / edges / params)
    complexity: Dict[str, int] = field(default_factory=dict)

    # 约束回执: {constraint_name: {"verdict": ..., "evidence": {...}}}
    guardrails: Dict[str, Any] = field(default_factory=dict)
    # 诊断: bin_errors / node_sens / loso ...
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # 溯源
    provenance: Dict[str, Any] = field(default_factory=dict)
    status: str = "proposed"
    born_at: float = field(default_factory=time.time)

    # ---- 便捷访问 ----
    @property
    def n_nodes(self) -> int:
        return int(self.complexity.get("nodes", 0))

    @property
    def n_params(self) -> int:
        return int(self.complexity.get("params", len(self.theta)))

    def is_admissible(self) -> bool:
        """是否全部约束 pass (无 violate)。"""
        for rec in self.guardrails.values():
            v = rec.get("verdict") if isinstance(rec, dict) else rec
            if v == "violate":
                return False
        return True

    def violations(self) -> List[str]:
        out = []
        for name, rec in self.guardrails.items():
            v = rec.get("verdict") if isinstance(rec, dict) else rec
            if v == "violate":
                out.append(name)
        return out

    # ---- 序列化 ----
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Candidate":
        known = {f for f in cls.__dataclass_fields__}          # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def compact(self, max_len: int = 400) -> str:
        """注入 prompt 用的紧凑文本视图 (≤max_len 字符).

        对应 SRHarness "runtime 保留详细证据但不塞进上下文"。
        全量证据仍在 archive 里, 需要时可重新纳入。
        """
        cplx = f"{self.n_nodes}n/{self.n_params}p"
        gl = ",".join(self.violations()) or "ok"
        s = (f"[{self.id}] loss={self.loss:.4g} cplx={cplx} guards={gl} "
             f":: {self.dsl}")
        return s if len(s) <= max_len else s[: max_len - 1] + "…"


class Archive:
    """append-only 候选档案.

    落盘格式: 每行一个 JSON 对象 (JSONL) —— 与既有 `runs_*/{name}.jsonl` 风格一致。
    - append 只追加, 不重写 → 可审计、可 diff、崩溃后重放即可恢复;
    - 内存索引 `_by_id` + 顺序 `_order` 加速查询;
    - `record()` 用于记录**非候选**事件 (工具调用 / 裁决 / 调度), 隔离在 `events`
      (不污染候选 Pareto 排序)。
    """

    def __init__(self, path: Optional[str] = None, *, task: str = "default"):
        self.path = path
        self.task = task
        self._by_id: Dict[str, Candidate] = {}
        self._order: List[str] = []
        self.events: List[Dict[str, Any]] = []
        if path and os.path.exists(path):
            self._replay()

    # ---------------- 持久化 ----------------
    def _replay(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue                                  # 半行 (崩溃残留) 跳过
                kind = rec.get("_kind", "candidate")
                if kind == "candidate":
                    c = Candidate.from_dict(rec.get("candidate", {}))
                    self._upsert(c, write=False)
                else:
                    self.events.append(rec)

    def _append(self, rec: Dict[str, Any]) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def _upsert(self, c: Candidate, write: bool = True) -> Candidate:
        """同 id 覆盖 (状态推进), 新 id 追加。持久化始终是"追加一条新记录"。"""
        if c.id not in self._by_id:
            self._order.append(c.id)
        self._by_id[c.id] = c
        if write:
            self._append({"_kind": "candidate", "ts": time.time(),
                          "task": self.task, "candidate": c.to_dict()})
        return c

    # ---------------- 写 ----------------
    def add(self, cand: Candidate) -> Candidate:
        """登记新候选 (或按 id 更新状态推进)."""
        if cand.id in self._by_id:
            raise ValueError(f"候选 id 已存在: {cand.id} (用 update())")
        return self._upsert(cand)

    def update(self, cand: Candidate) -> Candidate:
        """状态推进 (proposed→fitted→admitted...); 追加一条新记录以保留历史。"""
        if cand.id not in self._by_id:
            return self._upsert(cand)
        return self._upsert(cand)

    def record(self, event: str, **payload: Any) -> Dict[str, Any]:
        rec = {"_kind": "event", "ts": time.time(), "task": self.task,
               "event": event, **payload}
        self.events.append(rec)
        self._append(rec)
        return rec

    # ---------------- 读 ----------------
    def get(self, cid: str) -> Optional[Candidate]:
        return self._by_id.get(cid)

    def all(self, *, status: Optional[Iterable[str]] = None,
            admissible_only: bool = False) -> List[Candidate]:
        st = set(status) if status else None
        out = []
        for cid in self._order:
            c = self._by_id[cid]
            if st and c.status not in st:
                continue
            if admissible_only and not c.is_admissible():
                continue
            out.append(c)
        return out

    def ranked(self, *, reverse: bool = False, admissible_only: bool = False
               ) -> List[Candidate]:
        """按 loss 升序 (最好在前)."""
        cs = [c for c in self.all(admissible_only=admissible_only)
              if c.loss is not None and c.loss < float("inf")]
        return sorted(cs, key=lambda c: c.loss, reverse=reverse)

    def best(self, *, admissible_only: bool = False) -> Optional[Candidate]:
        r = self.ranked(admissible_only=admissible_only)
        return r[0] if r else None

    def __len__(self) -> int:
        return len(self._order)

    def __iter__(self):
        return iter(self.all())

    # ---------------- 回灌 ----------------
    @classmethod
    def from_legacy_runs(cls, output_dir: str, *, task: str = "legacy",
                         archive_path: Optional[str] = None,
                         glob: Optional[str] = None,
                         globs: Optional[Sequence[str]] = None) -> "Archive":
        """把既有跑批产物全量回灌为 archive (P0 验收第 1 步).

        兼容四种既有布局:
          - `{name}_final.json`: `{"best": {...}, "top_final": [...]}` (最老, 无 top);
          - `{name}_final.json`: `{"best": {...}, "top": [...]}`;
          - `summary_{scheme}.json`: `{"targets": {name: {dsl, loss, ...}}}`;
          - 单条记录 `{"dsl": ...}`。

        解析失败的条目记入 `events` 而不中断 (脏数据不应阻断回灌)。
        """
        import glob as _glob

        patterns = list(globs) if globs else [glob or "*_final.json"]
        ar = cls(archive_path, task=task)
        n_ok = n_bad = 0
        files: List[str] = []
        for pat in patterns:
            files.extend(_glob.glob(os.path.join(output_dir, "**", pat),
                                    recursive=True))
        for fp in sorted(set(files)):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception as e:                            # noqa: BLE001
                ar.record("legacy_replay_error", file=fp, error=repr(e))
                n_bad += 1
                continue
            recs = _extract_legacy_records(obj)
            vnames = []
            if isinstance(obj, dict):
                vnames = list((obj.get("profile") or {}).get("var_names") or [])
            for i, r in enumerate(recs):
                cplx = dict(r.get("complexity") or {})
                if not cplx:
                    cplx = _infer_complexity(str(r.get("dsl", "")))
                c = Candidate(
                    id=_new_id("L"),
                    dsl=str(r.get("dsl", "")),
                    var_names=list(r.get("var_names") or vnames),
                    theta=list(r.get("theta") or []),
                    loss=_as_float(r.get("loss")),
                    complexity=cplx,
                    metrics=dict(r.get("metrics") or {}),
                    provenance={"source": "legacy_replay", "file": fp,
                                "rank": i, "target": r.get("_target"),
                                "var_names": list(r.get("var_names") or vnames)},
                    status="fitted",
                )
                ar.add(c)
                n_ok += 1
        ar.record("legacy_replay_done", output_dir=output_dir,
                  patterns=patterns, n_ok=n_ok, n_bad=n_bad)
        return ar


def _as_float(v: Any) -> float:
    try:
        f = float(v)
        return f if f == f else float("inf")
    except (TypeError, ValueError):
        return float("inf")


def _infer_complexity(dsl: str) -> Dict[str, int]:
    """从 DSL 文本推断复杂度 (legacy 记录常缺 complexity 字段)。"""
    if not dsl:
        return {}
    try:
        from .core import dsl as _dsl
        # var_names 未知时用占位列表; parse_dsl 只按索引解析, 名字不影响计数
        g = _dsl.parse_dsl(dsl, [f"v{i}" for i in range(32)])
        c = g.complexity()
        return {"nodes": int(c.get("nodes", 0)), "edges": int(c.get("edges", 0)),
                "params": int(c.get("params", 0))}
    except Exception:                                        # noqa: BLE001
        return {"nodes": dsl.count("(") + 1}


def _extract_legacy_records(obj: Any) -> List[Dict[str, Any]]:
    """把各种既有 JSON 布局统一成"候选记录列表"。"""
    out: List[Dict[str, Any]] = []
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict) and "dsl" in x]
    if not isinstance(obj, dict):
        return out
    if "best" in obj or "top_final" in obj or "top" in obj:
        best = obj.get("best")
        if isinstance(best, dict) and "dsl" in best:
            out.append(best)
        for key in ("top_final", "top"):
            for r in (obj.get(key) or []):
                if isinstance(r, dict) and "dsl" in r:
                    out.append(r)
        return out
    if "targets" in obj and isinstance(obj["targets"], dict):
        for name, r in obj["targets"].items():
            if isinstance(r, dict) and "dsl" in r:
                r = dict(r)
                r["_target"] = name
                out.append(r)
        return out
    if "dsl" in obj:
        return [obj]
    return out
