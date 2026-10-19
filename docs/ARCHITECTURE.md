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

ASM 域内**两套口径并存**（同一内核与动作层，2026-10-19 起）：

| 口径 | 公式 | 模块 | 示例 |
|---|---|---|---|
| 单通道（旧） | `b23 = g(Θ)·T23` | `domain.py`（`ASMDomain`） | `examples/asm_b23/` |
| 多通道系数（新） | `b23 = Σ_j s_j·a_j(Θ)·T̃_j23` | `basisb2.py`（`BasisB2Domain`）+ `refit.py`（终口径） | `examples/asm_basisB2/` |

多通道口径的两个关键设计：

1. **一条系数 = 一个标准任务**。多通道不是单任务多目标，而是 J 条独立 SR 任务
   （`BasisB2Domain(target_index=j)`，13 个动作全链路可用）+ 终口径
   `refit.joint_refit` 把 J 条公式拼起来联合抛光。这正是报告的"SR 找形状 +
   联合定常数"两阶段法的落地。
2. **终口径可独立复算**。`refit.replay_run_refit` 从上层 `runs_basisB2/*/result.json`
   读冠军 DSL，在包内重做联合 LM —— 实测 e 0.0576 / g 0.1238 / m 0.0722 /
   v 2.2305 全部 rel 0.00%（含逐工况逐位一致）。框架自包含，数字不是引用报告。

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

**分析层 — `lgf_agent/analysis.py`**（只读查询，不写 archive）

状态层存在的第二个理由：让"事后一次性脚本"消失。历史脚本 `_tmp_t23_contrib.py`
为回答"最优公式有没有用张量基通道"而硬编码公式项、手工按 `Nv` 切分数组。现在：

| 函数 | 回答什么 | 是否需要数据 |
|---|---|---|
| `channel_of_var` | `T1_23` / `T5s_v` / `T10s_e` → 统一通道标签 | 否 |
| `tensor_channels_used` | 变量**池**里有哪些通道 | 否 |
| `vars_used_in_graph` | 公式**实际引用**了哪些变量 | 否 |
| `candidate_channel_usage` | 池口径 / 公式口径 / 解析是否成功，三栏分开 | 否 |
| `Archive.channel_usage()` | 一批候选的通道使用画像 | 否 |
| `leave_one_channel_out` | 单通道净贡献 RMS 占比 + 去掉后的 Δloss | 是 |

`leave_one_channel_out` 的定义：

```
contribution_c(x) = f(x) − f(x | vars_c := 0)
rms_share_c       = ‖contribution_c‖ / ‖f‖        （跨工况 pooled）
delta_loss_c      = loss(去掉 c) − loss(全式)      （loss = 逐工况 NMSE 再平均）
```

对任意图结构成立 —— 不要求线性求和，也不要求预先知道项的分组。通道"出现但无效"
（系数被拟合成 0）时 `rms_share` 自然趋近 0，这正是历史结论（T1s 净贡献仅 ~6%）
的机器化形式。**分类以"公式实际引用"为准**：池里有 T 通道而公式写成
`C1*sqrt(roloc)` 时该通道贡献按定义为 0，报出来只会制造噪声。

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
pytest -q                                                  # 57 项离线测试
python examples/asm_b23/run_p0_verify.py                   # 回灌 + 奇偶性机器判定 + 通道归因
```

`tests/` 的分布：

| 文件 | 覆盖 |
|---|---|
| `test_store_projector.py` | 候选契约、append-only 回放、四种 legacy 布局、脏数据容错、Pareto 去重与支配 |
| `test_verify.py` | 5 硬规则的正/反例、非数值约束降级、探针异常降级、汇总裁决 |
| `test_tools_agent.py` | 注册表、软错误、观测有界、成本账本、响应解析鲁棒性、离线端到端、时间预算 |
| `test_asm_domain.py` | 数据加载、度量口径、任务卡、**C1 奇偶性机器判定**、C1–C9 齐备、诱饵用例 |
| `test_basisb2_domain.py` | 多通道口径：RMS 归一/联合解量级/SR 权重对齐、任务卡与种子、**冠军 refit 独立复算（e 0.0576, rel 0.00%）**、PySR 内联常量参数化、demo 端到端两级口径、矩阵读取（上层工程不可用时自动 skip） |
| `test_analysis.py` | 三套通道命名解析、**LOCO 对已知真值的回收（<1e-6）**、公式未引用 vs 池无通道两个分支、archive 结构查询可重放、动作注册 |

## 6. 已知限制

- `agent_*` 字段只在 `agent_mode=True` 时生效；legacy 路径行为与旧版逐字一致。
- `code_exec` 默认关闭且未接真沙箱 —— 打开前需自行加固（限工作目录子路径、禁网、限时）。
- Inspector 的底座切换已留接口但未在实验中验证收益。
- 离线（heuristic）通道只能产出"结构种子级"质量的结果，用于 CI 与对照，**不代表**框架上限。
- 通道归因的**结构判定**对历史产物存在解析失败的情形（记录里的 `var_names` 不全，
  实测 2386 条中 149 条）：这些条目退回池口径归类，并在 `n_parse_failed` 里如实计数，
  不会被伪装成"公式没用通道"。
