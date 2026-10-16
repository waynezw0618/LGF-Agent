# 变更记录

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [0.1.0] — 2026-10-16

首个可发布骨架。LGF 升级为 LGF-Agent，五层架构 + 生成/审查分离，ASM 作为示例域。

### 新增 — L1 动作层 (`lgf_agent/tools.py`)
- 统一工具接口 `ToolResult(ok, observation, evidence, cost)`，其中 `observation` 是
  注入 prompt 的有界紧凑视图（≤400 字符），全量证据落 archive。
- 12 个科学动作：`profile_data` / `prior_check` / `fit_constants` / `evaluate_dsl` /
  `attribute` / `residual_view` / `guardrail_check` / `symbol_equiv` / `gp_search` /
  `submit_formula` / `read_skill` / `code_exec`（默认关闭）。
- `residual_view` 实现 SRHarness 消融中掉幅最大的"表达式视图"（残差统计 + 分箱误差，
  残差可作为下一轮拟合目标）。
- `gp_search` 把 GP 从主循环降格为一个动作；子代显式重置 `loss/theta`（修 V4b 的进化失效 bug）。

### 新增 — L2 状态层 (`lgf_agent/store.py`, `lgf_agent/projector.py`)
- `Candidate` 候选-证据契约（公式 / 参数 / 指标 / 约束回执 / 诊断 / 溯源 / 状态机）。
- `Archive` append-only JSONL：可审计、可 diff、崩溃后重放即恢复；`record()` 隔离记录非候选事件。
- `Archive.from_legacy_runs()` 兼容四种既有产物布局（`*_final.json` 的 `best`/`top_final`/`top`、
  `summary_*.json` 的 `targets`、单条记录），并保留 `var_names` 以便按口径分组。
- 投影 `pareto` / `topk` / `single_best`，含同公式去重、有界化与呈现顺序随机化
  （缓解 LLM 位置偏好）。

### 新增 — 约束层 (`lgf_agent/verify/`)
- `Constraint` / `ProbeResult` / `VerifyContext` / `Verifier` 协议：每条约束必须给出
  **可复现的数值证据**；`numeric=False` 的约束强制降级为 `unverified`。
- `default_hard_rules()` 泛化验收 5 硬规则中的 4 条：坍缩 / 爆参 / 跨工况稳定 / 值域护栏，
  另加目标级禁用算子。

### 新增 — L3/L4 (`lgf_agent/agent/`)
- `runtime.py`：动作序列解析（剥 reasoning + 去围栏 + 哨兵区分"解析失败/空对象"）、
  `Audit` 轨迹审计与断点续跑、`CostLedger` tokens/cost 逐调用会计。
- `roles.py`：`Creator`（生成 + 动作调用）与 `Inspector`（只裁决，数值探针为权威，
  LLM 只补充最小修复方向与待验证假设）。
- `scheduler.py`：R-C-L-K 显式化，三元预算硬上限，`Record(r,c,ℓ)` 全量 provenance。
- `skills.py`：领域知识卡加载（read_skill 的数据源）。

### 新增 — L0 领域层 (`lgf_agent/domains/`)
- `DomainAdapter` 协议 + `DatasetDomain` 最小实现。
- **ASM 示例域**：`b23 = g(I₁,I₂,…,Ro_loc,Re_t)·T23`，含 Pope 不变量语义、张量通道定义、
  数据集加载（兼容全量/精简两种 npz 布局）、**C1–C9 九条数值探针约束**、
  4 个假设族分支先验、2 张领域知识卡。
- `parity_audit()`：机器判定 T1/T5/T6/T10 的 23 分量为偶对称，并自动标记历史错误断言。
- `known_bad_assertions()`：记录已被机器判定为假的历史 physics context 断言。

### 新增 — 工程化
- 打包的 64 点/工况精简样例数据集（48.6 KB），示例与测试无需外部路径即可运行。
- 40 项离线测试（store/projector/verify/tools/agent/ASM 域）。
- `examples/asm_b23/`：离线 R-C-L-K 演示 + P0 判决点脚本。
- `docs/ARCHITECTURE.md`、`docs/LGF_Agent_方案.md`。

### 保留（行为不变）
- `core/pipeline_lgf.py`（legacy 双循环管线）、`core/gp.py`、`core/paramfit.py`、
  `core/compiler.py`、`core/dsl.py`、`core/graph.py`、`core/priors.py`、
  `core/attribution.py`、`core/eq_compat.py` —— 全部原样移植，未改动行为。
- `LGFConfig` 所有既有字段默认值不变；新增字段均为 `agent_*` 前缀且默认不改变 legacy 行为。
