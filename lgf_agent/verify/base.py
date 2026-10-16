"""lgf_agent.verify.base — 约束即数值探针 (protocol).

文献依据 (Tacke et al. 2026):
    "Whether a generated model satisfies the physical constraints is assessed
     **numerically**, by probing each constraint across a broad sample of
     deformation states, rotations, and perturbation directions."

我们的历史教训: 错误物理断言 (T1s "反对称") 以**文本**形式写进 prompt,
没有任何机制核对。所以本层的硬要求是 —— **每条约束必须给出可复现的数值证据**,
而不是一句 yes/no。

四种裁决:
    pass       证据支持约束成立
    violate    证据明确违反 (入池前拦截)
    warn       可疑但不致命 (记录, 不拦截)
    unverified 无法数值检验 → **标为假设, 不得当作约束使用** (方案设计原则 1)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

PASS, VIOLATE, WARN, UNVERIFIED = "pass", "violate", "warn", "unverified"


@dataclass
class ProbeResult:
    name: str
    verdict: str                              # pass | violate | warn | unverified
    evidence: Dict[str, Any] = field(default_factory=dict)
    severity: str = "medium"                  # low | medium | high
    min_fix: str = ""                         # 最小修复方向 (回执给 Creator)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(verdict=self.verdict, evidence=self.evidence,
                    severity=self.severity, min_fix=self.min_fix, note=self.note)


@dataclass
class VerifyContext:
    """一次裁决所需的全部上下文 (由域适配器构造)."""
    dsl: str
    var_names: Sequence[str]
    theta: Optional[Sequence[float]] = None
    cases: Optional[List[dict]] = None
    graph: Any = None                          # lgf_agent.core.graph.Graph
    domain: Any = None                         # DomainAdapter
    extras: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.extras.get(key, default)


class Constraint:
    """约束基类。子类实现 `check(ctx) -> ProbeResult`。

    `numeric=False` 表示该约束无法数值化 —— 基类会强制降级为 `unverified`,
    防止"把物理直觉当约束用"。
    """

    name: str = "unnamed"
    description: str = ""
    numeric: bool = True
    severity: str = "medium"

    def check(self, ctx: VerifyContext) -> ProbeResult:   # pragma: no cover
        raise NotImplementedError

    # 子类可覆盖: 该约束是否适用于给定上下文
    def applies(self, ctx: VerifyContext) -> bool:
        return True

    def run(self, ctx: VerifyContext) -> ProbeResult:
        if not self.numeric:
            return ProbeResult(
                self.name, UNVERIFIED, {"reason": "non_numeric_constraint"},
                self.severity,
                note="无法数值检验 → 标为待验证假设, 不得作为约束使用",
            )
        if not self.applies(ctx):
            return ProbeResult(self.name, PASS, {"skipped": "not_applicable"},
                               self.severity)
        try:
            r = self.check(ctx)
        except Exception as e:                                # noqa: BLE001
            return ProbeResult(self.name, UNVERIFIED,
                               {"error": repr(e)}, self.severity,
                               note="探针执行失败, 结果不可信")
        if not r.severity:
            r.severity = self.severity
        return r


class Verifier:
    """一组约束的编排器 (Inspector 的裁决内核)."""

    def __init__(self, constraints: Sequence[Constraint]):
        self.constraints: List[Constraint] = list(constraints)

    def register(self, c: Constraint) -> "Verifier":
        self.constraints.append(c)
        return self

    def run_all(self, ctx: VerifyContext) -> Dict[str, Dict[str, Any]]:
        return {c.name: c.run(ctx).to_dict() for c in self.constraints}

    @staticmethod
    def summarize(report: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        counts: Dict[str, int] = {PASS: 0, VIOLATE: 0, WARN: 0, UNVERIFIED: 0}
        violations, warnings, unverified = [], [], []
        for name, rec in report.items():
            v = rec.get("verdict", UNVERIFIED)
            counts[v] = counts.get(v, 0) + 1
            if v == VIOLATE:
                violations.append(name)
            elif v == WARN:
                warnings.append(name)
            elif v == UNVERIFIED:
                unverified.append(name)
        total = len(report) or 1
        return dict(counts=counts, violations=violations, warnings=warnings,
                    unverified=unverified, n=len(report),
                    pass_rate=counts[PASS] / total,
                    admissible=not violations)

    @staticmethod
    def verdict(report: Dict[str, Dict[str, Any]]) -> str:
        """汇总为 Inspector 的三态输出 (方案 §3.4)."""
        s = Verifier.summarize(report)
        if s["violations"]:
            return "reject"
        if s["warnings"] or s["unverified"]:
            return "return_for_refinement"
        return "pass"
