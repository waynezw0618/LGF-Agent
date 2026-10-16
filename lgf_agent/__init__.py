"""LGF-Agent — LLM 驱动的符号回归智能体框架.

定位
----
本包把 LLM-guided Graph-Feynman (LGF) 从"固定双循环管线 + 无状态 prompt 钩子"
升级为 **五层 + 三角色** 的智能体:

    L4 角色层   Creator(A) | Inspector(B) | Archivist(可选)   → agent/roles.py
    L3 轨迹层   R-C-L-K 调度 + provenance + 恢复 + 预算会计    → agent/scheduler.py
    L2 状态层   候选-证据契约 + append-only archive + 投影      → store.py / projector.py
                (+ 分析层: 通道归因 / 变量集画像)               → analysis.py
    L1 动作层   统一工具接口 (13 个可复用科学动作)              → tools.py
    L0 领域层   域适配器 (asm / ...)                            → domains/

设计原则
--------
1. **legacy 路径永久保留**: `core.pipeline_lgf.run_lgf` 一行不改, `cfg.agent_mode`
   选引擎, 两条路同预算可比 —— 这是"agent 比 legacy 好多少"这个结论的前提。
2. **诚实分开物理与建模**: 只裁决可数值检验的约束; 无法数值化的一律标
   `UNVERIFIED_ASSUMPTION`, 不得当作约束使用。
3. **不牺牲可审计性换性能**: 目标是物理合规 + 跨工况泛化 + 可追溯, 不是比 PySR 快。

快速开始
--------
    from lgf_agent import LGFConfig, run_agent
    from lgf_agent.domains.asm import ASMDomain

    dom = ASMDomain.from_dataset("outputs/asm/dataset_trueasm_g.npz")
    cfg = LGFConfig(agent_mode=True, agent_R=1, agent_C=1, agent_L=30, agent_K=1)
    result = run_agent(dom, cfg, llm_client=client)

参考
----
- SRHarness (Tsinghua FIB Lab, arXiv:2609.35501)
- Tacke et al., *LLM-driven design of physics-constrained constitutive models:
  two agents are better than one* (2026, DOI 10.1007/s44379-026-00099-3)
- Du et al., *LLMs for automatic equation discovery of nonlinear dynamics*,
  Phys. Fluids 36(9), 2024, DOI 10.1063/5.0224297
"""
from .version import __version__, __agent_protocol__
from .core.config import LGFConfig
from .store import Candidate, Archive
from .projector import project, PROJECTORS
from . import analysis
from .analysis import (channel_of_var, channel_groups, tensor_channels_used,
                       leave_one_channel_out, summarize_channel_usage)
from .tools import ToolRegistry, ToolResult, default_tools
from .verify import Constraint, ProbeResult, VerifyContext, Verifier
from .verify import default_hard_rules as hard_rules

__all__ = [
    "__version__", "__agent_protocol__",
    "LGFConfig",
    "Candidate", "Archive",
    "project", "PROJECTORS",
    "analysis", "channel_of_var", "channel_groups", "tensor_channels_used",
    "leave_one_channel_out", "summarize_channel_usage",
    "ToolRegistry", "ToolResult", "default_tools",
    "Constraint", "ProbeResult", "VerifyContext", "Verifier", "hard_rules",
    "run_agent",
]


def __getattr__(name):
    # 延迟导入: 避免 agent 子包在只用到 store/tools 时被强制加载 (及其 LLM 依赖)
    if name == "run_agent":
        from .agent.scheduler import run_agent
        return run_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
