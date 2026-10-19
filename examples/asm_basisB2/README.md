# 示例：方案B v2 — 多通道系数口径（basisB2）

对应报告 `outputs/asm/t23basisB2_paper/BasisB2_full_matrix_report.md`（2026-10-19）。
域代码在 `lgf_agent/domains/asm/basisb2.py` + `refit.py`。

## 口径

$$b_{23} = \sum_{j=1}^{J} s_j \, a_j(\Theta) \cdot \tilde T^{(j)}_{23}$$

- $J$ = 4（v/g/e 基：T1s/T5s/T6s/T10s）或 10（m 混合池）；
- $\tilde T^{(j)}_{23}$：通道 23 分量，**逐工况 RMS 归一**（消 3–4 个量级动态范围）；
- $\Theta = \{\eta_2, \zeta_2, S^2/\Omega^2, Re_\tau, r_{o,loc}, y_w\}$（yw∈[0,1]）；
- **两级口径**：中间（SR 逐通道独立估常数重组）/ 终（固定形状联合 LM refit）；
- 拟合权重 $w_i = \tilde U_{j,i}^2/(N_v\,\mathrm{var}_c)$ —— 系数误差逐点等于该通道
  对 b23 的 NMSE 贡献（此加权对**系数口径**正确；直接套 b23 会病态，10-16 已证）。

## 运行

```bash
# 0) （可选）从上层完整数据集生成精简 demo npz（64 点/工况）
python examples/asm_basisB2/pack_demo.py --src <上层>/outputs/asm

# 1) 离线跑通：stage base + 单系数 a_j 的 R-C-L-K + 两级口径演示（不调 LLM）
python examples/asm_basisB2/run_offline_demo.py --variant e --target 1

# 2) 回放历史冠军：独立复算终口径 refit，对账 0.0576 / 0.1238 / 0.0722
#    （需要上层完整数据集与 runs_basisB2；demo 数据会有 64≠129 点偏差）
python examples/asm_basisB2/run_refit_replay.py --data <上层>/outputs/asm

# 3) 汇总 runs_basisB2 复现报告 §3.1 矩阵（纯读取）+ 冠军公式通道归因
python examples/asm_basisB2/run_matrix_replay.py

# 4) P1 第一步：实测单题成本（GLM-5.3，key 在仓库 .env）→ 推 R-C-L-K 组合
python examples/asm_basisB2/probe_p1_cost.py --n 3

# 5) P1 主实验：三臂同预算对照（legacy vs agent±Inspector；先 heuristic 验证协议）
python examples/asm_basisB2/run_p1_arms.py --budget 60
python examples/asm_basisB2/run_p1_arms.py --budget 600 --backend openai   # 真 LLM
```

## 四个脚本各验证什么

| 脚本 | 验证 | 需要上层工程？ |
|---|---|---|
| `run_offline_demo.py` | 白化/靶标/加权口径正确；BasisB2Domain 13 动作全链路离线可跑；两级口径差可复现（refit 治愈抵消结构） | 否（demo 数据即可） |
| `run_refit_replay.py` | LGF-Agent **独立复算**终口径数字（e 0.0576 ±5%）——自包含性证明 | 是（完整数据 + runs） |
| `run_matrix_replay.py` | 报告 §3.1 的 27 组矩阵可从 result.json 机器重读 + 冠军 DSL 变量归因 | 是（runs） |
| `probe_p1_cost.py` | GLM-5.3 单题实测成本（tokens/秒/解析率）→ R-C-L-K 组合外推；strict 模式，429 即停不出假数据 | 否（key 即可） |
| `run_p1_arms.py` | 三臂同预算对照协议（同任务/同种子/同 time_budget_s）；heuristic 先验证，真 LLM 换 `--backend openai` | 否 |

## 与 asm_b23（旧单通道口径）的关系

`examples/asm_b23/` 保留 `b23 = g(Θ)·T23` 旧口径（P0 判决点示例仍在用）。
本目录是新口径——多通道 + 两阶段。两者共用同一内核与 13 个动作；
`refit.py` 是新口径独有的终口径组件。

## 已知结论（对账锚点，报告 §3–4）

- 总冠军 e_lgf_mg_big 终口径 **0.0576**（worst 0.068）= 标量 SOTA 0.0514 的 1.12×；
- 中间口径 10⁰ 级误差 84–92% 来自抵消结构（99–32201×），refit 回落 1–2 个量级；
- v 基不可行（cond 136–234）；m 池 legacy 爆炸可被 refit 治愈（0.126/0.125）；
- **in-sample 警告**：big 组（21–24 参）未做外推检验，48 参联合解外推已证不可行。
