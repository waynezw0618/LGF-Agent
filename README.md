# LGF-Agent

**LLM 驱动的符号回归智能体框架** —— 把 LGF（LLM-guided Graph-Feynman）从"固定双循环管线 + 无状态 prompt 钩子"升级为**五层架构 + 生成/审查分离**的可审计智能体。

```
┌──────────────────────────────────────────────────────────────┐
│ L4 角色层   Creator(A)  │  Inspector(B)                     │  agent/roles.py
├──────────────────────────────────────────────────────────────┤
│ L3 轨迹层   R-C-L-K 调度 + provenance + 恢复 + 预算会计        │  agent/scheduler.py
├──────────────────────────────────────────────────────────────┤
│ L2 状态层   候选-证据契约 · append-only archive · Pareto 投影  │  store.py / projector.py
│  （+ 分析层）通道归因 · 变量集画像 (只读查询)                  │  analysis.py
├──────────────────────────────────────────────────────────────┤
│ L1 动作层   13 个统一科学动作 (拟合/归因/残差视图/通道归因/...) │  tools.py
├──────────────────────────────────────────────────────────────┤
│ L0 领域层   asm (旋转槽道流 ASM 张量基闭合)  │  (未来域)       │  domains/
└──────────────────────────────────────────────────────────────┘
```

## 为什么需要它（三条已诊断的硬事实）

1. **动作空间被浪费**：原 `llm_ops.py` 注册了 5 个外层算子，实际只接线了 2 个（`G_INIT` / `M_GUIDE`），其余是死代码；而框架**本已具备**拟合、求值、先验、符号等价、GP 全套能力，只是没暴露给 LLM。所以动作层升级 80% 是"接线"。
2. **状态层缺失已经造成损失**：候选只活在内存里，跨进程即失。后果是"最优公式是否真用了 T23 通道"只能事后手写一次性脚本反算（`_tmp_t23_contrib.py`，硬编码公式项 + 手工按 `Nv` 切数组）；"LLM 干预到底有没有生效"完全不可观测；同族候选无法跨 run 统一排序。**现已修复**：`Archive.channel_usage()` 与 `analysis.leave_one_channel_out()` 取代了那个一次性脚本，`channel_attr` 成为第 13 个 L1 动作。
3. **没有审查者，错误物理断言直接进 prompt**：曾有"T1s 关于槽道中心反对称"的断言写进 prompt 若干轮无人核对 —— 实测为**偶对称**（even score ≈ +1.00）。本框架用 C1 数值探针机器裁决此类断言。

## 快速开始

```bash
pip install -e .            # 或: pip install -r requirements.txt
pytest -q                   # 57 项测试, 全部离线, 不需要 LLM key

# 1) 单通道口径: 离线 R-C-L-K + P0 判决点 (奇偶性机器判定 + 通道归因)
python examples/asm_b23/run_offline_demo.py
python examples/asm_b23/run_p0_verify.py

# 2) 多通道系数口径 (basisB2, 2026-10-19 报告轮):
python examples/asm_basisB2/pack_demo.py            # 一次性: 生成精简样例
python examples/asm_basisB2/run_offline_demo.py     # stage base + 单系数任务 + 两级口径
python examples/asm_basisB2/run_refit_replay.py --data <上层>/outputs/asm   # 复算冠军 0.0576
python examples/asm_basisB2/run_matrix_replay.py    # 报告 §3.1 矩阵重读
```

一站式回答"最优公式到底有没有用张量通道"（不再需要一次性脚本）：

```python
from lgf_agent import Archive, analysis

ar = Archive.from_legacy_runs("outputs/asm", globs=["*_final.json", "summary_*.json"])
print(ar.channel_usage())            # 公式口径 / 池口径 / 解析失败数, 三栏分开

# 单条公式的数值通道归因 (贡献 RMS 占比 + 去掉后的 Δloss)
rep = analysis.leave_one_channel_out(graph, theta, cases, var_names)
print(rep["ranking"])
```

接入真实 LLM：

```python
from lgf_agent import LGFConfig
from lgf_agent.agent.scheduler import run_agent
from lgf_agent.core.llm_ops import make_client
from lgf_agent.domains.asm import ASMDomain

client = make_client("openai", url="https://open.bigmodel.cn/api/paas/v4",
                     model="glm-5.3", api_key="...")

dom = ASMDomain.from_dataset("outputs/asm/dataset_trueasm_g.npz", variant="g")
cfg = LGFConfig(agent_mode=True, agent_R=1, agent_C=1, agent_L=30, agent_K=1,
                agent_projector="pareto", agent_inspector=True,
                agent_token_budget=2_000_000, agent_cost_budget_usd=5.0)

res = run_agent(dom, cfg, llm_client=client, out_dir="outputs")
print(res["best"]["dsl"], res["best"]["loss"], res["admit_ratio"])
```

## 示例域：旋转槽道流 ASM 张量基闭合（两套口径）

**多通道系数口径**（2026-10-19 报告轮，`examples/asm_basisB2/`）：

$$b_{23} = \sum_{j} s_j \, a_j(\Theta)\cdot \tilde T^{(j)}_{23}$$

- $J$=4（v/g/e 基：T1s/T5s/T6s/T10s）或 10（m 混合池）；通道逐工况 RMS 归一；
- $\Theta=\{\eta_2,\zeta_2,S^2/\Omega^2,Re_\tau,r_{o,loc},y_w\}$（yw∈[0,1]）；
- **两级口径**：中间（SR 逐通道估常数重组）/ 终（固定形状联合 LM refit，
  `refit.py` 可独立复算报告冠军 e **0.0576** / m 0.0722 / g 0.1238，rel 0.00%）；
- 拟合权重 $w=\tilde U_j^2/(N_v\,\mathrm{var}_c)$：系数误差逐点等于该通道对 b23 的 NMSE 贡献。

**单通道口径**（旧，`examples/asm_b23/`）：$b_{23}=g(\Theta)\cdot T_{23}$，P0 判决点示例仍在用。

两口径共用同一内核、13 个动作与 C1–C9 探针。域内提供 **C1–C9 九条数值探针约束**：
奇偶对称 / 张量可表性 / 值域 / 全局常数 / LOSO / 量纲 / 数值鲁棒 / 复杂度 / 反作弊。

## 设计原则

1. **legacy 路径永久保留** —— `core.pipeline_lgf.run_lgf` 一行不改，`cfg.agent_mode` 选引擎。两条路同预算可比，才谈得上"agent 比 legacy 好多少"。
2. **诚实分开物理与建模** —— 只裁决可数值检验的约束；无法数值化的一律标 `UNVERIFIED_ASSUMPTION`，不得当约束使用。
3. **不牺牲可审计性换性能** —— 目标是物理合规 + 跨工况泛化 + 可追溯，不是比 PySR 快。

## 目录结构

```
lgf_agent/
  core/          域无关内核 (graph/compiler/dsl/priors/gp/paramfit/attribution/...)
                 + legacy pipeline_lgf (原样保留)
  tools.py       L1 动作层: ToolRegistry + 13 个科学动作
  store.py       L2 状态层: Candidate 契约 + append-only Archive (+ channel_usage 查询)
  projector.py   L2 投影: pareto / topk / single_best
  analysis.py    L2 分析: 通道命名解析 / 结构判定 / leave-one-channel-out 数值归因
  verify/        约束层: Constraint 协议 + 5 硬规则泛化
  agent/         L3/L4: runtime / roles(Creator,Inspector) / scheduler(R-C-L-K) / skills
  domains/       L0: base 协议 + asm 示例域 (单通道 + basisB2 多通道系数口径)
examples/asm_b23/      单通道口径示例 (+ 打包样例数据, P0 判决点)
examples/asm_basisB2/  多通道系数口径示例 (demo 打包 / 离线任务 / 冠军复算 / 矩阵重读)
tests/                 57 项离线测试
docs/                  架构说明与升级方案
```

## 参考文献

- SRHarness — Tsinghua FIB Lab, arXiv:2609.35501（可复用科学动作 / 持久科学状态 / R-C-L-K 轨迹生命周期）
- Tacke et al., *LLM-driven design of physics-constrained constitutive models: two agents are better than one*, 2026, DOI 10.1007/s44379-026-00099-3
- Du et al., *Large language models for automatic equation discovery of nonlinear dynamics*, Phys. Fluids 36(9), 2024, DOI 10.1063/5.0224297
- Pope, S. B. (1975). *A more general effective-viscosity hypothesis*. JFM 72:331
- Weatheritt & Sandberg (2016). JCP 325:22-37
- Yang & Wang (2018). JFM 838

## 许可

MIT。见 `LICENSE`。
