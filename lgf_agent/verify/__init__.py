"""lgf_agent.verify — 约束层 (数值探针).

对外接口:
    Constraint / ProbeResult / VerifyContext / Verifier   (base.py)
    default_hard_rules / 各硬规则类                        (hard_rules.py)

域专属约束 (如 ASM 域的 C1 奇偶对称 / C2 张量可表性 / C5 LOSO / C9 反作弊)
在 `lgf_agent.domains.<name>.constraints` 中实现, 复用同一套协议。
"""
from .base import (PASS, VIOLATE, WARN, UNVERIFIED, Constraint, ProbeResult,
                   VerifyContext, Verifier)          # noqa: F401
from .hard_rules import (CollapseRule, CrossCaseStabilityRule, ForbiddenOpRule,
                         ParamBlowupRule, RangeGuardRule, default_hard_rules,
                         compile_and_eval, per_case_nmse)  # noqa: F401

__all__ = [
    "PASS", "VIOLATE", "WARN", "UNVERIFIED",
    "Constraint", "ProbeResult", "VerifyContext", "Verifier",
    "CollapseRule", "ParamBlowupRule", "CrossCaseStabilityRule",
    "RangeGuardRule", "ForbiddenOpRule", "default_hard_rules",
    "compile_and_eval", "per_case_nmse",
]
