"""lgf_agent.domains.base — 域适配器协议 (L0).

域适配器是"把某个具体物理问题接到 LGF-Agent 上"的唯一接口。已有的先例证明
这条路可行: `shipman/seeds.py` 的 monkeypatch 手法已在证明"领域层可以零改内核接入"。

**关键设计 (方案 §3.4 与 P3)**:
物理上下文必须拆成两类字段 ——
- `verified_assertions()`: 可由 Inspector 数值探针**核实**的断言 → 进 prompt 作为约束;
- `unverified_assumptions()`: 无法数值检验的物理直觉 → **标注为假设**, 不得作为约束。

历史教训: `rc_asm_vw_t23_full.py:125-126` 把 "T1s 关于槽道中心反对称" 当事实写进
prompt, 实测是**偶对称** (even≈+1.00), 没有任何机制核对。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


class DomainAdapter:
    """域适配器基类 (鸭子类型即可, 不强制继承)。"""

    name: str = "domain"
    description: str = ""

    # ---- 数据 ----
    def build_cases(self, cfg) -> Tuple[List[dict], List[str]]:
        """返回 (cases, var_names)。

        cases: list of {'X': (N, n_vars), 'y': (N,), 'name': str,
                        [+ optional 'w': (N,) 加权]}
        """
        raise NotImplementedError

    # ---- 约束 ----
    def constraints(self, cfg) -> List[Any]:
        """返回该域的 Constraint 列表 (数值探针)。"""
        return []

    # ---- prompt 材料 ----
    def task_card(self) -> str:
        """任务卡: 注入每次 Creator prompt 的任务说明 (可与 branch 无关)."""
        return self.description

    def branch_priors(self, cfg) -> List[str]:
        """C 个分支各自的**假设族先验** (必须声明这是假设而非事实)。"""
        return []

    def verified_assertions(self) -> List[str]:
        """可被数值探针核实的物理断言 (进 prompt 作为约束)。"""
        return []

    def unverified_assumptions(self) -> List[str]:
        """无法数值化的物理直觉 (标为假设, 不得当约束)。"""
        return []

    def skills(self, include: Optional[Sequence[str]] = None) -> Dict[str, str]:
        """领域知识卡 {topic: markdown}。"""
        return {}

    # ---- 早停判据 ----
    def target_met(self, cand) -> bool:
        """「提交公式通过全部约束且跨工况 worst-case 达标」的域判据。"""
        return False


class DatasetDomain(DomainAdapter):
    """从内存数据构造的最小域 (用于测试与临时实验)。"""

    def __init__(self, cases: List[dict], var_names: Sequence[str], *,
                 name: str = "dataset", description: str = "",
                 constraints_fn=None, branch_priors_fn=None,
                 target_met_fn=None, skills: Optional[Dict[str, str]] = None):
        self._cases = cases
        self._vars = list(var_names)
        self.name = name
        self.description = description or f"通用数据集域 ({len(cases)} 工况)"
        self._constraints_fn = constraints_fn
        self._branch_fn = branch_priors_fn
        self._target_fn = target_met_fn
        self._skills = skills or {}

    def build_cases(self, cfg):
        return self._cases, self._vars

    def constraints(self, cfg):
        return list(self._constraints_fn(cfg)) if self._constraints_fn else []

    def branch_priors(self, cfg):
        return list(self._branch_fn(cfg)) if self._branch_fn else []

    def skills(self, include=None):
        if not include:
            return dict(self._skills)
        return {k: v for k, v in self._skills.items() if k in set(include)}

    def target_met(self, cand) -> bool:
        return bool(self._target_fn(cand)) if self._target_fn else False
