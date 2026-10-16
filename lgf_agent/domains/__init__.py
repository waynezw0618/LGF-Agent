"""lgf_agent.domains — L0 领域层.

每个子包 = 一个物理问题的适配器 (实现 `domains.base.DomainAdapter` 协议)。
现有: `asm` (旋转槽道流 ASM 张量基闭合, 本次升级的第一落地域)。

新增一个域只需三件东西:
1. `build_cases(cfg)` → 把该域的原始数据变成 (cases, var_names);
2. `constraints(cfg)` → 该域的数值探针约束 (复用 verify 协议);
3. `task_card()` / `branch_priors()` / `skills()` → 给 LLM 的域知识。

**无需改动内核**: 既有 `shipman/seeds.py` 的 monkeypatch 手法已证明这条路可行。
"""
from .base import DomainAdapter, DatasetDomain   # noqa: F401

__all__ = ["DomainAdapter", "DatasetDomain"]
