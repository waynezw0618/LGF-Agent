"""lgf_agent.core — 域无关内核 (自 LGF 移植, 行为保持不变).

模块
----
graph       方程 = 带边特征的 DAG
compiler    图 → sympy/lambdify 可求值对象 (+ 解析梯度)
dsl         LGF-DSL 解析 / 打印
templates   结构模板库 (饱和/指数饱和/Voce/对数等)
priors      数据画像 + AI-Feynman 风格结构检验 (对称/可分/组合/量纲)
gp          图空间遗传算子 (crossover/mutate/select/extinction)
paramfit    多启动 L-BFGS-B 联合拟合 (解析梯度 + 硬 deadline + 值域护栏)
attribution 节点级归因 (敏感度 + 形状标签 + 表达式视图原料)
eq_compat   符号等价判定 / 等式文本 → LGF-DSL
config      LGFConfig (含 agent_* 字段)
llm_ops     LLM 后端 (heuristic/openai 兼容) + 老外层算子 (G_INIT/M_GUIDE 等)
benchmarks  合成基准 (冒烟/回归测试用)
pipeline_lgf  legacy 双循环管线 (保留原样, agent_mode=False 时使用)
"""
from . import graph          # noqa: F401
from . import compiler       # noqa: F401
from . import dsl            # noqa: F401
from . import priors         # noqa: F401
from . import paramfit       # noqa: F401

__all__ = ["graph", "compiler", "dsl", "priors", "paramfit"]
