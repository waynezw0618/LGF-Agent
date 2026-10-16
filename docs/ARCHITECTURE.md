# LGF-Agent 架构说明

> 面向实现者。设计动机与文献依据见同目录 `LGF_Agent_方案.md`。

## 1. 数据流总览

```
                     ┌────────────────────────────────────────┐
     DomainAdapter   │  build_cases()  →  (cases, var_names)   │
     (L0)            │  constraints()  →  [Constraint, ...]    │
                     │  task_card() / branch_priors() / skills │
                     └───────────────┬────────────────────────┘
                                     │
   scheduler.run_agent  ┌───────────▼───────────┐
   (L3)                 │  ToolContext          │  cases, var_names, cfg,
                        │  + Verifier           │  archive, extras, skills, rng
                        │  + ToolRegistry (L1)  │
                        └───────────┬───────────┘
                                    │
        ┌───────────────────────────┼────────────────────────────┐
        │  for r in R:              │                            │
        │    pool = project(S, ρ)   │  ← L2 投影 (pareto/topk/single_best)
        │    for c in C:            │                            │
        │      for ℓ in L:          │                            │
        │        Creator.propose()  │──▶ 动作序列 + 候选          │
        │        Inspector.review() │──▶ 数值探针裁决             │
        │        Archive.add/update │──▶ append-only 落盘          │
        └───────────────────────────┴────────────────────────────┘
```

## 2. 各层职责与不变量

### L0 领域层 — `lgf_agent/domains/`

**唯一契约**：把某个物理问题翻译成 `(cases, var_names)`，并声明该域的约束与知识。

| 方法 | 作用 | 不变量 |
|---|---|---|
| `build_cases(cfg)` | 返回 `cases`（`{'X','y','name'[,'w']}`）与 `var_names` | `X.shape[1] == len(var_names)`；`y` 长度与 `X` 一致 |
| `constraints(cfg)` | 该域的 `Constraint` 列表 | 每条约束必须可数值检验（或显式声明 `numeric=False`） |
| `task_card()` | 注入 Creator prompt 的任务卡 | **只能包含可核实的事实** |
| `verified_assertions()` / `unverified_assumptions()` | 物理上下文二分 | 前者交由探针裁决，后者禁止作为约束 |
| `branch_priors(cfg)` | C 个分支的假设族先验 | 措辞必须声明"这是假设" |

> **为什么强制二分**：曾有错误断言（T1s 反对称）以"事实"身份进入 prompt 并无机制核对。
> 现在凡写进 `verified_assertions()` 的内容都必须有一条探针能证伪它。

### L1 动作层 — `lgf_agent/tools.py`

```python
@dataclass
class ToolResult:
    ok: bool
    observation: str      # ≤400 字符, 注入 prompt
    evidence: dict        # 全量机器可读证据, 落 archive
    cost: dict            # {seconds, tokens, calls}
    payload: dict         # 调度器内部使用, 不注入 prompt
```

`ToolRegistry.dispatch(name, **args)` 是唯一入口：解析失败 / 参数不匹配 / 工具异常
一律转成 `ok=False` 的软错误（Agent 循环不应因单次工具失败而中断），同时记入 `trace` 与成本账本。

**观测压缩**是硬约束：`evidence` 可以很大，`observation` 必须小。这是"runtime 保留详细证据
但不塞进上下文"的落地。

### L2 状态层 — `lgf_agent/store.py` + `projector.py`

`Archive` 的写入语义是**只追加**：

- 新候选 → `add()` → 追加一条 `{"_kind":"candidate", ...}`；
- 状态推进（`proposed→fitted→admitted/rejected→submitted`）→ `update()` → **再追加一条**，
  内存索引覆盖，磁盘保留历史；
- 非候选事件（工具调用、裁决、调度步）→ `record()`，隔离在 `events`，**不污染 Pareto 排序**。

恢复：`Archive(path)` 构造时自动重放；半行（崩溃残留）被跳过而不抛错。

`projector.project(cands, view=..., max_n=..., shuffle=...)`：

- `pareto`：`(loss ↓, complexity ↓)` 非支配前沿，同 `dsl` 先去重；
- `topk` / `single_best`：次要信号 / 文献消融对照臂；
- `shuffle` 只打乱顺序不改入选集合（缓解位置偏好）。

### L3 轨迹层 — `lgf_agent/agent/scheduler.py`

R-C-L-K 的四元组语义：

| 参数 | 含义 | 默认 | 消融事实（SRHarness, 基线 74.47%） |
|---|---|---|---|
| `agent_R` | 重启轮数：每轮用 TopK 历史重建上下文 | 1 | R2C1L15 → 66.67%（更差） |
| `agent_C` | 独立分支数：每分支赋不同假设族先验 | 1 | R1C2L15 → 72.07%（更差） |
| `agent_L` | 单分支精炼步数 | 30 | **同样预算下加深精炼最优** |
| `agent_K` | 每步局部采样数（best-of-K） | 1 | R1C1L15K2 → 73.87% |

三元预算硬上限：`time_budget_s`、`agent_token_budget`、`agent_cost_budget_usd`；
任一耗尽即 `stop_reason` 记录并退出，已产出结果保留。

### L4 角色层 — `lgf_agent/agent/roles.py`

**Creator**：把领域任务卡 + 工具清单 + `P_ρ(S)` 候选池视图 + 分支先验 + 上一步 observation
组装成 prompt，要求 LLM 返回 `{thought, actions, candidates}`。
`materialize()` 负责把 DSL 变成候选：拟合 → 过全部数值探针 → 定 `status` → 落 archive。

**Inspector**：`review(cand)` 先取（或现算）数值探针报告，`Verifier.verdict()` 给出三态：

- 有 `violate` → `reject`
- 有 `warn` 或 `unverified` → `return_for_refinement`
- 全 `pass` → `pass`

**数值裁决是权威**。LLM 只在 `source="numeric+llm"` 时补充 `min_fix` 与
`unverified_assumptions`，**不能改判**。

> 换底座（`agent_inspector_backend` 等）留给 P4 对照臂。理由：一上来就把"有没有 Inspector"
> 和"换不换底座"两个变量混在一起，实验就说不清了。

## 3. 评分与"优化的是不是 b23"

工具层的损失函数来自 `core.compiler.eval_cases`，它有两个分支：

- **无 `w`**（默认）：`loss = mean_c[ mean_y(res²) / var_c(y) ]` —— 即逐工况 NMSE 再平均，
  与项目统一口径 `loss_vw` 逐字一致；
- **有 `w`**：加权 NMSE，权重语义是"把逐点误差折算到某个物理量上"。

> ⚠️ 域适配器给 `cases` 加 `w` 时必须自带推导说明。ASM 域早期版本把
> `T̃23²/(Nv·var_c)` 直接套在未归一的 `b23` 上，引入 1.8×10⁸ 倍的动态范围，
> 参数拟合立刻病态。现在该口径只在 `weighting="channel"` 且与白化通道配套时启用。

## 4. 扩展一个新域

```python
from lgf_agent.domains.base import DomainAdapter
from lgf_agent.verify import Constraint, ProbeResult

class MyDomain(DomainAdapter):
    name = "mydomain"

    def build_cases(self, cfg): ...            # (cases, var_names)
    def constraints(self, cfg):  return [MyRule()]
    def task_card(self):         return "..."
    def branch_priors(self, cfg): return ["...", "..."]
    def skills(self, include=None): return {"my_card": open(...).read()}
    def target_met(self, cand):  return cand.loss < 0.05 and cand.is_admissible()
```

然后 `run_agent(MyDomain(), cfg, llm_client=client)` 即可 —— **无需改动内核**。

## 5. 测试与复现

```bash
pytest -q                                    # 40 项离线测试
python examples/asm_b23/run_p0_verify.py     # 回灌 + 奇偶性机器判定
```

`tests/` 的分布：

| 文件 | 覆盖 |
|---|---|
| `test_store_projector.py` | 候选契约、append-only 回放、四种 legacy 布局、脏数据容错、Pareto 去重与支配 |
| `test_verify.py` | 5 硬规则的正/反例、非数值约束降级、探针异常降级、汇总裁决 |
| `test_tools_agent.py` | 注册表、软错误、观测有界、成本账本、响应解析鲁棒性、离线端到端、时间预算 |
| `test_asm_domain.py` | 数据加载、度量口径、任务卡、**C1 奇偶性机器判定**、C1–C9 齐备、诱饵用例 |

## 6. 已知限制

- `agent_*` 字段只在 `agent_mode=True` 时生效；legacy 路径行为与旧版逐字一致。
- `code_exec` 默认关闭且未接真沙箱 —— 打开前需自行加固（限工作目录子路径、禁网、限时）。
- Inspector 的底座切换已留接口但未在实验中验证收益。
- 离线（heuristic）通道只能产出"结构种子级"质量的结果，用于 CI 与对照，**不代表**框架上限。
