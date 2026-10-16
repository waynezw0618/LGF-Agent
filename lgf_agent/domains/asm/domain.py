"""lgf_agent.domains.asm.domain — 旋转槽道流 ASM 域适配器.

**当前口径 (2026-10-16, 仍在开发中)**

    b23(y) = g( I₁, I₂, …, Ro_loc, Re_t ) · T₂₃

其中:
- `I_i` 为 **Pope (1975) 完整性基** 的 5 个不变量 (eta2/zeta2/s3/so2/s2o2);
- `Ro_loc`、`Re_t` 为局部分辨标量扩展 (旋转数 / 湍流雷诺数);
- `T₂₃` 为张量基通道的 23 分量 (b23 展开中只有 T1/T5/T6/T10 的 23 分量非零);
- `g` 是待发现的标量函数 (只能含全局常数)。

物理背景: 旋转槽道流, Re_τ = 180, Ro_τ ∈ {7.5, 15, 30, 75, 150}, 系统旋转沿流向 x₁。
参考: Yang & Wang 2018 JFM 838; Weatheritt & Sandberg 2016 JCP 325:22-37;
      Pope 1975 JFM 72:331; Gatski & Speziale 1993 JFM 254:59.

> ⚠️ **诚实声明 (方案设计原则 2)**: 本域把物理上下文严格拆成两类 ——
> `verified_assertions()` (可由 C1..C9 数值探针核实, 进 prompt 作约束) 与
> `unverified_assumptions()` (无法数值化, 标为假设, **不得**当约束用)。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..base import DomainAdapter
from . import dataset as ds_mod
from .channels import (CHANNEL_DESC, CHANNEL_PARITY, COMP6, INDEX23,
                       NONZERO_CHANNELS_23, channel_legend)
from .constraints import asm_constraints, even_score
from .invariants import (DEFAULT_VARS, LOCAL_EXTENSIONS, POPE_INVARIANTS,
                         WALL_EXTENSIONS, invariant_legend)

_SKILL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")


class ASMDomain(DomainAdapter):
    name = "asm"
    description = ("旋转槽道流代数应力模型 (ASM) 张量基闭合: "
                   "b23 = g(I1,I2,...,Ro_loc,Re_t) · T23")

    def __init__(self, path: str, *, variant: str = "g",
                 var_mode: str = "minimal", channel_mode: str = "single",
                 channel_norm: str = "pooled", with_wall: bool = True,
                 weighting: str = "nmse",
                 loso_probe: bool = False,
                 decoy_components: Sequence[int] = (0, 3, 4),
                 target_loss: float = 0.07, seed: int = 0):
        self.path = path
        self.variant = variant
        self.var_mode = var_mode
        self.channel_mode = channel_mode
        self.channel_norm = channel_norm
        self.with_wall = with_wall
        self.weighting = weighting
        self.loso_probe = loso_probe
        self.decoy_components = tuple(decoy_components or ())
        self.target_loss = float(target_loss)
        self.seed = seed
        self._loaded: Optional[Dict[str, Any]] = None
        self.extras_cache: Dict[str, Any] = {}

    # ---------------- 构造 ----------------
    @classmethod
    def from_dataset(cls, path: str, **kw) -> "ASMDomain":
        return cls(path, **kw)

    @classmethod
    def from_outputs(cls, outputs_dir: str = "outputs/asm", variant: str = "g",
                     **kw) -> "ASMDomain":
        return cls(os.path.join(outputs_dir, f"dataset_trueasm_{variant}.npz"),
                   variant=variant, **kw)

    # ---------------- 数据 ----------------
    def _ensure(self) -> Dict[str, Any]:
        if self._loaded is None:
            self._loaded = ds_mod.load_cases(
                self.path, variant=self.variant, channel_mode=self.channel_mode,
                var_mode=self.var_mode, channel_norm=self.channel_norm,
                with_wall=self.with_wall, weighting=self.weighting,
                decoy_components=self.decoy_components)
            self.extras_cache = {"case_extras": self._loaded["extras"],
                                 "decoy_cases": self._loaded.get("decoys") or {}}
        return self._loaded

    def build_cases(self, cfg) -> tuple:
        d = self._ensure()
        return d["cases"], d["var_names"]

    def evaluate(self, preds: Sequence[np.ndarray]) -> Dict[str, Any]:
        d = self._ensure()
        return ds_mod.evaluate_b23_nmse(preds, d["extras"])

    def extras(self) -> Dict[str, Any]:
        self._ensure()
        return self.extras_cache

    @property
    def var_names(self) -> List[str]:
        return list(self._ensure()["var_names"])

    # ---------------- 约束 ----------------
    def constraints(self, cfg) -> List[Any]:
        return asm_constraints(cfg, loso_probe=self.loso_probe)

    # ---------------- prompt 材料 ----------------
    def task_card(self) -> str:
        d = self._ensure()
        vn = ", ".join(d["var_names"])
        return f"""[任务] 旋转槽道流 ASM 张量基闭合 (Re_τ=180, Ro_τ=7.5/15/30/75/150, 旋转沿流向)。
目标形式 (当前口径): **b23 = g(Θ) · T23**, 其中 Θ = 下列自变量, T23 为张量基通道的 23 分量
(跨工况常数, **已知, 无需拟合**)。

自变量 Θ = [{vn}]:
{invariant_legend()}

张量通道 (b23 展开中非零的四条):
{channel_legend()}

评估口径: loss_vw = mean_c[ mean_y (b23_pred - b23)² / var_c(b23) ] (逐工况 NMSE 再平均)。
注意: 权重已按 w = T̃23²/(Nv·var_c) 设定, 故"加权系数误差"**逐点等于** b23 的误差贡献 ——
优化 loss 即优化 b23, 不要额外引入 T23 的近似。

硬规则: 只用**全局常数 C#** (5 工况通用); 严禁逐工况常数 K#。
"""

    def branch_priors(self, cfg) -> List[str]:
        """C 个分支各自的**假设族先验** (必须声明这是假设, 不是事实)."""
        return [
            "F1 阻尼族 (假设): 近壁阻尼主导, 骨架形如 C·exp(-yw/τ)、1/(C+yw)、"
            "1/(1+(yw/τ)²)。依据: 壁面种子在既有实验中一致有效。",
            "F2 对数律族 (假设): 形如 √roloc·(C·log(yw)+C'·roloc) 的可分离乘积。"
            "依据: V4b 冠军公式所在族 (yw→0 发散须配饱和/旋转因子)。",
            "F3 张量基族 (假设): 系数按不变量展开 a(Θ)=Σ c_k φ_k(Θ) 的有理/饱和组合。"
            "依据: 10-16 结构化联合拟合 g 基 0.0530 / e 基 0.0482。",
            "F4 机制族 (假设): 涡黏类比 ν_t·S₂₃ 结构, 即 g 与 √(eta2) 或 eta2 的"
            "单调关系 + 旋转修正因子。依据: 物理可解释性优先。",
        ]

    def verified_assertions(self) -> List[str]:
        """可由 C1..C9 数值探针核实的断言 (进 prompt 作约束)."""
        return [
            "b23 关于槽道中心 y→-y 为**偶对称** (C1 探针核实)。",
            f"b23 展开中只有 T{NONZERO_CHANNELS_23} 的 23 分量非零, "
            f"且它们与 b23 同为偶对称 (C1 核实)。",
            "真值 b23 全 y 恒正、偏置主导 (mean/std 1.55-2.91), Ro 非单调 (Ro30 最大) "
            "(C3 核实)。",
            "5 工况必须共享同一组全局常数 (C4 核实)。",
            "自变量全部为无量纲量 (Pope 不变量 / Re_t / Ro_loc / yw) (C6 核实)。",
        ]

    def unverified_assumptions(self) -> List[str]:
        """无法数值化的物理直觉 → 标为假设, 不得当约束. """
        return [
            "高旋转下流动准二维化, 因此系数应出现饱和结构 (tanh/exp(-·)/比值) "
            "—— 这是物理解释, 非可核实断言。",
            "近壁低 Re_t 阻尼的 Myong-Kasagi f_μ 型形式 —— 类比假设。",
            "涡黏比拟 ν_t ∝ k/ε 的适用性 —— 在本流动族未被独立验证。",
        ]

    def skills(self, include: Optional[Sequence[str]] = None) -> Dict[str, str]:
        from ...agent.skills import load_skill_dir
        return load_skill_dir(_SKILL_DIR, include=include)

    def seed_skeletons(self, branch: int = 0) -> List[str]:
        """离线 / 冷启动用的参数化骨架 (对应该分支的假设族)。

        注意: 这些是**结构种子**, 参数仍由 paramfit 拟合; 它们不是"答案",
        只是保证无 LLM 时管线也能产出有意义候选, 便于 CI 冒烟与 legacy 对照。
        """
        fam = [
            # F1 阻尼族
            ["C1*(exp(inv(C2*(yw)+C3*(1))))",
             "C1*(inv(C2*(yw)+C3*(1)))",
             "C1*(inv(C2*(yw)^2+C3*(1)))"],
            # F2 对数律族
            ["C1*(sqrt(roloc))*(C2*(log(yw))+C3*(roloc))",
             "C1*(sqrt(roloc))+C2*(roloc)*(log(C3*(yw)))",
             "C1*(log(yw+ C2*(1)))*(roloc)^2"],
            # F3 张量基族 (有理/饱和)
            ["C1*(eta2)*(inv(C2*(1)+C3*(eta2)))+C4*(roloc)*(inv(C5*(1)+C6*(roloc)))",
             "C1*(tanh(C2*(roloc)))*(log(yw+C3*(1)))",
             "C1*(s2o2)*(inv(C2*(1)+C3*(zeta2)))"],
            # F4 机制族 (涡黏类比)
            ["C1*(sqrt(eta2))*(inv(C2*(1)+C3*(roloc)))",
             "C1*(eta2)*(inv(C2*(1)+C3*(zeta2)))",
             "C1*(ret)^2*(inv(C2*(1)+C3*(ret)^2))"],
        ]
        k = int(branch) % len(fam)
        out = list(fam[k])
        # 其它族的 1 个代表, 保证跨分支有重叠也有差异
        for j in range(len(fam)):
            if j != k:
                out.append(fam[j][0])
        return out

    # ---------------- 断言审计 (P0 判决点第 3 条) ----------------
    def parity_audit(self) -> Dict[str, Any]:
        """机器判定各通道 23 分量的奇偶性, 并与 CHANNEL_PARITY 声明对账。

        这是 P0 的关键验收: 应复现 `even_score ≈ +1.00` (偶对称), 从而**自动标记**
        physics context 中"T1s 反对称"那条错误断言。
        """
        import numpy as np
        d = self._ensure()
        ds = ds_mod.load_npz(self.path)
        out: Dict[str, Any] = {"channels": {}, "claim": {}, "consistent": True}
        for m in NONZERO_CHANNELS_23:
            key = f"c0_T{m}"
            if key not in ds.files:
                continue
            valid = ds["c0_valid"].astype(bool)
            yh = ds["c0_yh"][valid].astype(float)
            arr = np.asarray(ds[key])
            vals = (arr[valid, INDEX23] if arr.ndim == 2 else arr[valid]).astype(float)
            sc = even_score(yh, vals)
            out["channels"][f"T{m}_23"] = round(sc, 4)
        valid = ds["c0_valid"].astype(bool)
        yh0 = ds["c0_yh"][valid].astype(float)
        out["channels"]["b23"] = round(
            even_score(yh0, ds["c0_b6"][valid, INDEX23].astype(float)), 4)
        # 对照: 一个已知的**奇**函数 (yh 自身) 应给出 ≈ -1, 证明探针有区分力
        out["control_odd_yh"] = round(even_score(yh0, yh0), 4)
        for m, claim in CHANNEL_PARITY.items():
            name = f"T{m}_23"
            if name in out["channels"]:
                got = out["channels"][name]
                ok = (got > 0.6) if claim == "even" else (got < -0.6)
                out["claim"][name] = {"declared": claim, "even_score": got,
                                      "verified": bool(ok)}
                out["consistent"] = out["consistent"] and bool(ok)
        return out

    def known_bad_assertions(self, repo_root: Optional[str] = None) -> List[Dict[str, Any]]:
        """历史错误断言清单 (已被机器判定为假)。

        这些断言曾以"事实"形式写进 prompt。列出它们是为了:
        (1) 让审稿人看到框架能捕获自己过去的错误;
        (2) 防止回归 —— 若有人再写进去, 可以拿这份清单自动比对。
        """
        root = repo_root or os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
        f = os.path.join(root, "scripts", "rc_asm_vw_t23_full.py")
        rec = {"assertion": "T1s = S₂₃ = ½dW̄/dy 关于槽道中心**反对称**",
               "verdict": "FALSE", "source_file": f,
               "source_lines": "125-126",
               "machine_evidence": "C1 探针: even_score(T1_23) ≈ +1.00 (偶对称)",
               "found_in_file": os.path.exists(f)}
        return [rec]

    # ---------------- 早停 ----------------
    def target_met(self, cand) -> bool:
        """提交公式通过全部约束 且 跨工况 worst-case 达标。"""
        if not cand.is_admissible():
            return False
        loss = float(cand.loss or float("inf"))
        if loss > self.target_loss:
            return False
        nms = [v for k, v in (cand.metrics or {}).items() if k.endswith("_nmse")]
        if nms and max(nms) / (min(nms) + 1e-12) > 5.0:
            return False
        return True
