"""lgf_agent.config — 配置入口 (转出 core.config 的 LGFConfig).

保留这一层是为了让 `from lgf_agent.config import LGFConfig` 与
`from lgf_agent import LGFConfig` 都成立, 且不破坏 core 内 legacy 代码的相对导入.
"""
from .core.config import LGFConfig  # noqa: F401

__all__ = ["LGFConfig"]
