# 示例：旋转槽道流 ASM 张量基闭合

本目录演示如何把 **LGF-Agent** 接到一个真实物理问题上。域适配器代码在
`lgf_agent/domains/asm/`。

## 当前口径

$$b_{23}(y) = g\!\left(I_1, I_2, \ldots, \mathrm{Ro}_{loc}, \mathrm{Re}_t\right)\cdot T_{23}$$

- $I_i$：**Pope (1975) 完整性基**的 5 个不变量 —— `eta2=tr(S²)`、`zeta2=tr(W²)`、
  `s3=tr(S³)`、`so2=tr(SW²)`、`s2o2=tr(S²W²)`
- `Ro_loc = 2Ωk/ε`、`Re_t = k²/(νε)`：局部分辨标量扩展
- `yw = 1-|y/h|`：无量纲真壁距（非局部量，但本工程证明确需）
- $T_{23}$：张量基通道的 23 分量（跨工况常数，已知，无需拟合；b23 展开中只有
  T1/T5/T6/T10 的 23 分量非零）

> ⚠️ **口径状态：开发中。** 变量集、通道归一与评估权重仍在演进；历史产物跨多个口径，
> 比较时必须按变量集分组（`run_p0_verify.py` 会自动做这件事）。

## 数据

- `data/dataset_trueasm_g_demo.npz` —— **打包的精简样例**（64 点/工况，48.6 KB）。
  足以跑通全部示例与测试。
- 完整数据集（129 点/工况，含 T1..T10 六分量）来自上层工程的
  `outputs/asm/dataset_trueasm_{v,g,e}.npz`。把 `--dataset` 指过去即可，代码无需改动。

> 原始 DNS：Yang & Wang 2018 JFM 838，Re_τ = 180，Ro_τ ∈ {7.5, 15, 30, 75, 150}，
> 系统旋转沿流向 x₁。三套变量（v / g / e）的差异在 W 的定义（Gatski-Speziale 修正与否）。

## 运行

```bash
# 1) 离线跑通 R-C-L-K（不调 LLM）
python examples/asm_b23/run_offline_demo.py

#    常用开关：--C 2 --K 2 --projector single_best --no-inspector
#    这些正是 P4 消融臂；--no-inspector 是"单 agent"对照。

# 2) P0 判决点：回灌历史产物 + 机器判定奇偶性断言 + 通道归因
python examples/asm_b23/run_p0_verify.py --legacy <上层工程>/outputs/asm
```

`run_p0_verify.py` 做四件事（对应升级方案 §4 的 P0 验收）：

1. **C1 奇偶性机器判定** —— 对 `T1_23 / T5_23 / T6_23 / T10_23 / b23` 计算 even score
   （应 ≈ +1.00），并用一个已知奇函数作对照（应 ≈ −1）证明探针有区分力；
   同时自动标记历史 physics context 里的错误断言。
2. **archive 回灌** —— 把 `outputs/asm/runs_*/**/*_final.json` 与 `summary_*.json`
   全量灌进 append-only archive（本次实测 274 文件 → 2386 候选）。
3. **投影** —— 按**变量集分组**算 Pareto 前沿，直接复现历史排序。
4. **通道归因** —— 直接回答"最优公式到底有没有用张量基通道"：
   - **结构查询**（零依赖）：2386 条里公式口径 80 条真用了通道、2306 条纯标量；
     在"变量池含 10 个通道"的 18 维 m 池分组（162 条）中，**按 loss 排序的前 8 名
     全部是零通道式**（纯 `roloc`/`yw`）—— 历史结论被机器复现。
   - **数值分解**（`analysis.leave_one_channel_out`）：把某通道输入置零后重算，
     得到 `rms_share`（净贡献 RMS 占比）与 `delta_loss`（去掉后的代价）。
     用一个已知真值的合成算例（`y = 2√roloc + 0.5·T1`）校准，与解析值一致到 < 1e-6。
   - 这一步正式取代了以前的一次性脚本 `_tmp_t23_contrib.py`。

> 判据：若第 1 步不能自动跑出来，说明约束层设计有问题 —— 不要进入 P1。

> 示例产物默认落在仓库根的 `run/`（可用 `LGF_AGENT_OUTDIR` 覆盖），
> 不再写进包内 `outputs/`，以免与上层工程的真实跑批产物混淆。

## 接真实 LLM

```bash
export LGF_LLM_API_KEY=<你的 key>
```

```python
from lgf_agent import LGFConfig
from lgf_agent.agent.scheduler import run_agent
from lgf_agent.core.llm_ops import make_client
from lgf_agent.domains.asm import ASMDomain

client = make_client("openai", url="https://open.bigmodel.cn/api/paas/v4",
                     model="glm-5.3", api_key="...", max_tokens=16384,
                     extra_body={"thinking": {"type": "enabled"},
                                 "reasoning_effort": "low"})

dom = ASMDomain.from_dataset(".../dataset_trueasm_g.npz", variant="g")
cfg = LGFConfig(agent_mode=True, agent_R=1, agent_C=1, agent_L=30, agent_K=1,
                agent_projector="pareto", agent_inspector=True,
                agent_token_budget=2_000_000, agent_cost_budget_usd=5.0,
                time_budget_s=600.0)
res = run_agent(dom, cfg, llm_client=client, out_dir="outputs")
```

## 产物布局

```
outputs/agent/<task>/
  archive.jsonl   候选-证据契约 (append-only; 可重放/可 diff)
  traj.jsonl      调度步 + 工具调用 + 裁决 (provenance, 断点续跑依据)
  result.json     best / top / admit_ratio / stop_reason / cost
```

`result.json` 里的 `admit_ratio` = 入池候选通过全部约束的比率，
对标 Tacke et al. 2026 的 91%→100% 口径。

## 九条数值探针约束（C1–C9）

| # | 约束 | 探针在做什么 |
|---|---|---|
| C1 | 奇偶对称性 | 对 y→−y 镜像后按 \|yh\| 分箱比较左右两侧 |
| C2 | 张量形式可表性 | T 基投影残差 + 基共线性 cond + 抵消比 |
| C3 | 值域 | 预测符号一致性 / 量级带（真值 b23 恒正、偏置主导） |
| C4 | 跨工况常数共享 | 图中是否含逐工况常数 K# |
| C5 | 跨 Ro 泛化 | LOSO mean/worst（需 `loso_probe=True`） |
| C6 | 量纲一致性 | 自变量是否都在无量纲白名单内 |
| C7 | 数值鲁棒性 | 极端输入扰动下的 NaN/inf/定义域错误 |
| C8 | 复杂度 | 节点/参数上限 |
| C9 | 反作弊 | 在诱饵目标（uu/uv/uw 分量）上是否异常好 |
