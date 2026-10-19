"""lgf_agent.domains.asm.basisb2 — 方案B v2: 多通道系数口径 + 白化 + 两阶段.

口径 (与上层工程 scripts/rc_asm_basisB2.py 逐字对齐, 2026-10-18/19 报告轮)
---------------------------------------------------------------------
目标形式 (比旧的单通道 b23=g(Θ)·T23 更进一步):

    b23 = Σ_{j=1..J} s_j · a_j(Θ) · Ũ_j

- 张量通道逐工况 RMS 归一 (cnorm="percase", 报告主口径), 再按加权 Gram 矩阵
  **白化** (whiten, 截断 λ ≤ tol·λ_max) 得 Ũ;
- 报告轮实际用的白化配置: `--whiten 0`(关闭) + `--cnorm percase` —— 即 Ũ 就是
  RMS 归一通道本身, J=4 (v/g/e) 或 J=10 (m 池), whiten 保留为可选路径;
- a_j(Θ) 是待发现的系数函数, Θ = BASE = [eta2, zeta2, s2o2, ret, roloc, yw];
- s_j = 通道 j 的 β 靶标 RMS (把 a_j 归一到 O(1)), 靶标 b_j/|s_j| 即 SR 任务 y。

**两级口径** (报告 §1, 测试必须区分):
- 中间口径: SR 逐通道独立拟常数后直接重组 —— 方法对比用;
- 终口径 (joint refit): 固定 SR 公式形状, 全部通道常数联合 LM 最小化重组 NMSE
  (见 refit.py) —— PySR 内联常量同样参数化后同场抛光, 两方法公平。

拟合权重 (为什么这里加权是**对的**, 而旧 b23 直接加权是错的):
    w_i = Ũ_{j,i}² / (Nv · var_c(b23))
对**系数** a_j 的加权 NMSE 逐点等于该通道对 b23 的 NMSE 贡献 —— 前提是目标是
系数 (本口径), 而不是 b23 本身 (旧口径, 会引入 1.8e8 动态范围病态, 10-16 已证)。

两阶段 (stage):
- base: 结构化联合加权 lstsq, a_j(Θ)=Σ_k c_jk φ_k(Θ), 给出基线精度 + β_j 逐点
  靶标 (POOL 见下方, bounded pool 保证 |β|≲O(5));
- sr:   对每条 β_j(Θ) 做符号回归 (LGF-Agent 的 13 个动作 / legacy / PySR)。

已知结论 (报告 §3-4, 作为本域的 sanity 参照):
- e_lgf_mg_big 终口径 0.0576 (标量 SOTA 0.0514 的 1.12×), worst 0.068;
- 中间口径 10⁰ 级误差 84-92% 来自"逐通道独立估常数 × 抵消结构"
  (抵消比 99-32201×), refit 后回落 1-2 个量级;
- v 基不可行 (T 通道共线 cond 136-234, 逐工况自由系数残差仍 ~1.06);
- m 池 legacy 爆炸 (99-117) 可被 refit 治愈 (0.126/0.125)。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .channels import INDEX23, NONZERO_CHANNELS_23

RO_CASES: List[float] = [7.5, 15.0, 30.0, 75.0, 150.0]

# Θ 自变量 (固定顺序, 与上层 BASE 逐字一致; refit.py 依赖此顺序)
BASE_VARS: List[str] = ["eta2", "zeta2", "s2o2", "ret", "roloc", "yw"]

# 系数基池 (bounded, 全部基 |φ| ≲ O(10), 保证 β O(1)) — "rich" 池
# (上层 POOL; compact/unbounded 池见上层脚本, 本域默认 rich)
COEF_POOL: List[Tuple[str, Any]] = None  # 延迟构造 (需要 numpy f-字典协议)


def coef_pool() -> List[Tuple[str, Any]]:
    """Θ 系数基池 (name, fn(f)->array)。f 为 {BASE_VARS: array} 字典。"""
    return [
        ("1", lambda f: np.ones_like(f["yw"])),
        ("sqrt(rl)/(1+sqrt(rl))",
         lambda f: np.sqrt(f["roloc"]) / (1 + np.sqrt(f["roloc"]))),
        ("tanh(roloc)", lambda f: np.tanh(f["roloc"])),
        ("roloc/(1+roloc)", lambda f: f["roloc"] / (1 + f["roloc"])),
        ("eta2/(1+eta2)", lambda f: f["eta2"] / (1 + f["eta2"])),
        ("zeta2/(1+zeta2)", lambda f: f["zeta2"] / (1 + f["zeta2"])),
        ("s2o2/(1e4+|s2o2|)",
         lambda f: f["s2o2"] / (1e4 + np.abs(f["s2o2"]))),
        ("log((eta2+.01)/(eta2+1))",
         lambda f: np.log((f["eta2"] + 1e-2) / (f["eta2"] + 1))),
        ("log1p(ret)", lambda f: np.log1p(f["ret"])),
        ("yw", lambda f: f["yw"]),
        ("yw^2", lambda f: f["yw"] ** 2),
        ("yw*sqrt(rl)/(1+sqrt(rl))",
         lambda f: f["yw"] * np.sqrt(f["roloc"]) / (1 + np.sqrt(f["roloc"]))),
    ]


# m 池通道名 (T1 共享 + T5/T6/T10 × {v,g,e})
M_POOL_CHANNELS: List[str] = ["T1s"] + [
    f"{t}s_{v}" for t in ("T5", "T6", "T10") for v in ("v", "g", "e")]
NAMED_CHANNELS: Dict[str, List[str]] = {
    "v": ["T1s", "T5s", "T6s", "T10s"],
    "g": ["T1s", "T5s", "T6s", "T10s"],
    "e": ["T1s", "T5s", "T6s", "T10s"],
    "m": M_POOL_CHANNELS,
}


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def _t23(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """兼容两种 npz 布局: 全量 (N,6) → 取 23 分量; 精简 (N,) 直接用。"""
    arr = np.asarray(arr)
    return (arr[valid, INDEX23] if arr.ndim == 2 else arr[valid]).astype(float)


def load_channels(variant: str, paths: Dict[str, str], *,
                  cnorm: str = "percase") -> List[Dict[str, Any]]:
    """加载 5 工况的 (Θ, b23, T 通道矩阵), 通道逐工况 RMS 归一 (报告主口径)。

    paths: {variant: npz 路径}; m 池需要 {"v":..,"g":..,"e":..} 三套。
    兼容两种 npz 布局 (与 dataset.py 相同): (N,6) 全量 / (N,) 精简 23 分量。
    """
    if variant == "m":
        for v in ("v", "g", "e"):
            if v not in paths:
                raise FileNotFoundError(f"m 池需要 dataset_trueasm_{{v,g,e}}.npz, 缺 {v}")
        dss = {v: np.load(paths[v], allow_pickle=True) for v in ("v", "g", "e")}
    else:
        if variant not in paths:
            raise FileNotFoundError(f"缺 {variant} 数据集")
        dss = {variant: np.load(paths[variant], allow_pickle=True)}

    cases = []
    for i, ro in enumerate(RO_CASES):
        p = f"c{i}_"
        if variant == "m":
            src = dss["v"]
            valid = src[p + "valid"].astype(bool)
            yh = src[p + "yh"][valid].astype(float)
            sc = src[p + "scalars"][valid].astype(float)
            b23 = src[p + "b6"][valid, INDEX23].astype(float)
            cols = []
            for t in ("T1", "T5", "T6", "T10"):
                srcs = ("v",) if t == "T1" else ("v", "g", "e")
                for v in srcs:
                    cols.append(_t23(dss[v][p + t], valid))
            ch = np.column_stack(cols)
        else:
            ds = dss[variant]
            valid = ds[p + "valid"].astype(bool)
            yh = ds[p + "yh"][valid].astype(float)
            sc = ds[p + "scalars"][valid].astype(float)
            b23 = ds[p + "b6"][valid, INDEX23].astype(float)
            ch = np.column_stack(
                [_t23(ds[p + f"T{m}"], valid) for m in NONZERO_CHANNELS_23])
        f = dict(eta2=sc[:, 0], zeta2=sc[:, 1], s2o2=sc[:, 4], ret=sc[:, 5],
                 roloc=sc[:, 6], yw=1.0 - np.abs(yh))
        if cnorm == "percase":
            rr = np.sqrt(np.mean(ch ** 2, 0))
            rr[rr <= 0] = 1.0
            ch = ch / rr
        cases.append(dict(ro=ro, yh=yh, b23=b23, f=f, ch=ch, Nv=len(yh),
                          var_c=float(np.var(b23))))
    return cases


def weight_vector(cases: Sequence[Dict[str, Any]]) -> np.ndarray:
    """逐工况 NMSE 权重 1/(Nv·var_c), 拼接 (上层 weight())。"""
    return np.concatenate([
        np.full(c["Nv"], 1.0 / (c["Nv"] * c["var_c"])) for c in cases])


def whiten(cases: List[Dict[str, Any]], jmax: Optional[int] = None,
           tol: float = 0.0, use_whiten: bool = False):
    """加权 Gram 白化 (可选)。返回 (W, lam, J), 各 case 写入 'U' 键。

    报告轮默认 use_whiten=False (Ũ = RMS 归一通道), 保留白化为可复现实验路径。
    """
    X = np.vstack([c["ch"] for c in cases])
    w = weight_vector(cases)
    G = X.T @ (X * w[:, None])
    if not use_whiten:
        lam = np.diag(G).copy()
        for c in cases:
            c["U"] = c["ch"]
        return np.eye(X.shape[1]), lam, X.shape[1]
    lam_, V = np.linalg.eigh(G)
    o = np.argsort(lam_)[::-1]
    lam_, V = lam_[o], V[:, o]
    lam_[lam_ < 0] = 0.0
    keep = lam_ > tol * lam_[0]
    J = int(min(int(keep.sum()), jmax or keep.sum()))
    lam_k, V_k = lam_[:J], V[:, :J]
    W = V_k / np.sqrt(lam_k)[None, :]
    for c in cases:
        c["U"] = c["ch"] @ W
    return W, lam_, J


# ---------------------------------------------------------------------------
# stage base: 结构化联合解 + β 靶标
# ---------------------------------------------------------------------------
def wlstsq(Z: np.ndarray, y: np.ndarray, w: np.ndarray,
           ridge: float = 0.0) -> np.ndarray:
    """加权最小二乘 (列归一化); ridge>0 时岭回归取最小范数解。"""
    sw = np.sqrt(w)
    Zs, ys = Z * sw[:, None], y * sw
    sc = np.sqrt((Zs ** 2).sum(0))
    sc[sc <= 0] = 1.0
    Zn = Zs / sc
    if ridge > 0:
        A = Zn.T @ Zn
        A[np.diag_indices_from(A)] += ridge
        cs = np.linalg.solve(A, Zn.T @ ys)
    else:
        cs, *_ = np.linalg.lstsq(Zn, ys, rcond=1e-11)
    return cs / sc


def eval_nmse(pred_list: Sequence[np.ndarray],
              cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """重组 b23 的逐工况 NMSE (loss_vw 口径, 单一来源)。"""
    pc, nms = {}, []
    for p, c in zip(pred_list, cases):
        r = float(np.mean((np.asarray(p) - c["b23"]) ** 2) / c["var_c"])
        nms.append(r)
        pc[f"Ro{c['ro']:g}"] = round(r, 4)
    return dict(loss_vw=float(np.mean(nms)), worst=float(np.max(nms)),
                per_case=pc)


def stage_base(cases: List[Dict[str, Any]], *,
               pool: Optional[Sequence[Tuple[str, Any]]] = None,
               ridge: float = 0.0, use_whiten: bool = False,
               jmax: Optional[int] = None, tol: float = 0.0,
               train_ros: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """结构化联合加权 lstsq: a_j = Σ_k c_jk φ_k → β_j 逐点靶标 + 基线精度。

    返回 dict: X/U/b23/var_c/ro 逐工况数组 + coef 矩阵 + summary (loss 等)。
    这是 "stage base" 的纯函数版 (不落盘), 调用方决定持久化方式。
    """
    pool = list(pool) if pool is not None else coef_pool()
    whiten(cases, jmax=jmax, tol=tol, use_whiten=use_whiten)
    K = len(pool)
    Zs, names = [], []
    for c in cases:
        B = np.column_stack([fn(c["f"]) for _, fn in pool])          # Nv×K
        J = c["U"].shape[1]
        Zs.append((B[:, :, None] * c["U"][:, None, :]).reshape(c["Nv"], K * J))
    names = [f"{kn}·U{j + 1}" for kn, _ in pool for j in range(cases[0]["U"].shape[1])]
    tr = [i for i, c in enumerate(cases)
          if (train_ros is None or c["ro"] in train_ros)]
    Zall = np.vstack([Zs[i] for i in tr])
    yall = np.concatenate([cases[i]["b23"] for i in tr])
    wall = weight_vector([cases[i] for i in tr])
    coef = wlstsq(Zall, yall, wall, ridge)
    preds = [Zs[i] @ coef for i in range(len(cases))]
    r = eval_nmse(preds, cases)
    J = cases[0]["U"].shape[1]
    cmat = coef.reshape(K, J).T                                   # cmat[j,k]

    tgt: Dict[int, List[np.ndarray]] = {}
    for j in range(J):
        tgt[j] = [np.column_stack([fn(c["f"]) for _, fn in pool]) @ cmat[j]
                  for c in cases]
    beta_rms = [float(np.sqrt(np.mean(np.concatenate(tgt[j]) ** 2)))
                for j in range(J)]
    return dict(cases=cases, design=Zs, coef_flat=coef, coef=cmat,
                targets=tgt, beta_rms=beta_rms, K=K, J=J,
                pool=[n for n, _ in pool], loss_vw=r["loss_vw"],
                worst=r["worst"], per_case=r["per_case"])


def sr_cases_for_target(base_out: Dict[str, Any], j: int, *,
                        train_ros: Optional[Sequence[float]] = None
                        ) -> Tuple[List[dict], float]:
    """把第 j 条 β 靶标转成 SR 拟合用 cases (X=Θ, y=β_j/s_j, w=Ũ_j²/(Nv·var_c))。

    返回 (cases, scale)。与上层 _cases_for_target 逐字对齐:
    scale = pooled RMS(β_j over train cases), y 归一到 |y|≲O(1)。
    """
    cases = base_out["cases"]
    tr = [i for i, c in enumerate(cases)
          if (train_ros is None or c["ro"] in train_ros)]
    pooled = np.concatenate([base_out["targets"][j][i] for i in tr])
    scale = float(np.sqrt(np.mean(pooled ** 2))) or 1.0
    out = []
    for i, c in enumerate(cases):
        X = np.column_stack([c["f"][k] for k in BASE_VARS])
        Uj = c["U"][:, j]
        var_c = c["var_c"] + 1e-300
        nv = c["Nv"]
        out.append(dict(i=i, ro=c["ro"], name=f"Ro{c['ro']:g}",
                        X=X, y=base_out["targets"][j][i] / scale,
                        w=Uj ** 2 / (nv * var_c), Nv=nv))
    return out, scale


def reconstruct(preds_per_j: Sequence[Sequence[np.ndarray]],
                scales: Sequence[float],
                cases: Sequence[Dict[str, Any]]) -> List[np.ndarray]:
    """重组 b23 = Σ_j scale_j · a_j(Θ) · Ũ_j (中间口径评估用)。"""
    out = []
    for i, c in enumerate(cases):
        acc = np.zeros(c["Nv"])
        for j in range(len(scales)):
            acc = acc + np.asarray(preds_per_j[j][i]) * scales[j] * c["U"][:, j]
        out.append(acc)
    return out


# ---------------------------------------------------------------------------
# 域适配器
# ---------------------------------------------------------------------------
class BasisB2Domain:
    """方案B v2 域: 一次拟合**一条**系数 a_j(Θ) (j 由 target_index 指定)。

    多通道是"J 条独立 SR 任务 + 联合 refit"的组合, 不是单任务多目标 ——
    这正是两阶段设计的意义: 每条 a_j 是一个标准 LGF-Agent 任务 (13 动作全可用),
    终口径由 refit.joint_refit 把 J 条公式拼起来联合抛光。

    用法:
        base = stage_base(load_channels("e", paths))          # 一次
        dom  = BasisB2Domain(base, target_index=0)            # a_1 任务
        cases, var_names = dom.build_cases(cfg)               # X=Θ, y=β_1/s_1
    """

    name = "asm_basisb2"
    description = ("方案B v2: b23 = Σ_j s_j·a_j(Θ)·T̃_j23 (多通道系数口径, "
                   "逐工况 RMS 归一, 两级: SR 中间口径 + joint refit 终口径)")

    def __init__(self, base_out: Dict[str, Any], *, target_index: int = 0,
                 variant: str = "g", target_loss: float = 0.10,
                 seed: int = 0):
        if not (0 <= target_index < base_out["J"]):
            raise IndexError(f"target_index={target_index} 超出 J={base_out['J']}")
        self.base = base_out
        self.j = int(target_index)
        self.variant = variant
        self.target_loss = float(target_loss)
        self.seed = seed
        self._cases, self._scale = sr_cases_for_target(base_out, self.j)
        self.chan_name = NAMED_CHANNELS.get(variant, [f"U{j+1}" for j in
                                                      range(base_out["J"])])
        self._cached: Dict[str, Any] = {}

    # ---- 数据 ----
    def build_cases(self, cfg) -> Tuple[List[dict], List[str]]:
        return self._cases, list(BASE_VARS)

    @property
    def var_names(self) -> List[str]:
        return list(BASE_VARS)

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def channel_name(self) -> str:
        return self.chan_name[self.j] if self.j < len(self.chan_name) else f"U{self.j+1}"

    def evaluate(self, preds: Sequence[np.ndarray]) -> Dict[str, Any]:
        """该通道系数误差 → 对 b23 的 NMSE 贡献 (加权口径下逐点相等)。"""
        per, nms = {}, []
        for p, c in zip(preds, self._cases):
            w = c["w"]
            sw = w.sum()
            var_w = float(np.dot(w, (c["y"] - np.dot(w, c["y"]) / sw) ** 2) / sw) + 1e-12
            r = float(np.dot(w, (np.asarray(p) - c["y"]) ** 2) / sw) / var_w
            nms.append(r)
            per[f"Ro{c['ro']:g}"] = round(r, 4)
        return {"loss_vw": float(np.mean(nms)), "worst": float(np.max(nms)),
                "per_case": per, "scale": self._scale,
                "channel": self.channel_name}

    def extras(self) -> Dict[str, Any]:
        cases = self.base["cases"]
        if not self._cached:
            from .dataset import CaseExtra
            self._cached = {
                "case_extras": [
                    CaseExtra(ro=c["ro"], yh=c["yh"], b23=c["b23"],
                              T23=c["U"][:, 0], T23n=c["U"][:, 0],
                              var_c=c["var_c"], channels=c["U"],
                              channel_names=list(self.chan_name))
                    for c in cases],
                "basisb2": {"variant": self.variant, "J": self.base["J"],
                            "channel_names": list(self.chan_name),
                            "beta_rms": list(self.base["beta_rms"]),
                            "base_loss_vw": self.base["loss_vw"],
                            "target_index": self.j,
                            "scale": self._scale},
            }
        return self._cached

    # ---- 约束 ----
    def constraints(self, cfg) -> List[Any]:
        from .constraints import asm_constraints
        return asm_constraints(cfg, loso_probe=False)

    # ---- prompt 材料 ----
    def task_card(self) -> str:
        ex = self.extras()["basisb2"]
        M = self.j + 1
        ch = self.channel_name
        return f"""[任务] 旋转槽道流 ASM 张量基闭合 (方案B v2, 系数口径)。
Re_τ=180, Ro_τ=7.5/15/30/75/150 (系统旋转沿流向)。

目标形式: **b23 = Σ_j s_j·a_j(Θ)·T̃_j23** (J={ex['J']} 条张量通道)。
本次拟合**第 {M} 条系数 a_{M}(Θ)**, 对应通道 {ch};
T̃ 通道为 23 分量经**逐工况 RMS 归一** (量级 O(1), 已知, 无需拟合);
a_{M} 已按 s_{M}={self._scale:.4g} 归一到 |a|≲O(1)。

Θ (自变量, 逐点局部量, 5 工况共用):
- eta2 = tr(S²) ≥ 0        (Pope I1; 1e-6 → 3e2, 近壁极小, 对数变化)
- zeta2 = tr(W²) ≥ 0       (Pope I2; {self.variant} 基中含 Ω 贡献)
- s2o2 = tr(S²W²)           (Pope 不变量; ~2e4 量级)
- ret = k²/(νε)             (局部湍流雷诺数; 6e-5 → 5e4)
- roloc = 2Ωk/ε             (局部旋转数; 6e-5 → 15.4, 区分工况的核心变量)
- yw = 1-|y/h| ∈ [0,1]      (真壁距; 0=壁面, 1=槽道中心; 非局部壁面扩展)

通道物理角色 (决定 a_{M} 的先验形状):
- T1 (S 通道, 平均剪切): a1 是"涡黏型"系数 — 主要由 eta2/ret/壁面 yw 决定,
  对 roloc 弱依赖; 旋转不直接进入。
- T5/T6/T10 (W̄/Ω 通道): "旋转响应"系数 — 强依赖 roloc, 常见 tanh(C·roloc)、
  C/(C'+roloc)、exp(-C·roloc)、√roloc 饱和/衰减族。
- 近壁族: eta2/ret 跨 9 个量级 → log(eta2+eps)、1/(C+eta2)、eta2/(C+eta2)、
  exp(-C·yw)、yw、yw²; 切忌 eta2 的大幂 (会爆)。
- 有理形式 g1 = C1/(1+C2·eta2+…) 是 Speziale 类系数的标准形状。

结构化联合解的系数结构 (48 参数加权 lstsq 上界, 供参考起点, 可自由改进):
{self._structure_hint()}

评估口径: 加权 NMSE, w = Ũ_{M}²/(Nv·var_c(b23)) — 系数误差逐点等于该通道
对 b23 的 NMSE 贡献, 优化 loss 即优化 b23。
两级口径: 本任务 loss 是**中间口径** (逐通道独立); 终口径 = 固定形状联合 refit
(所有通道常数一起抛光), 中间口径 10⁰ 级误差 84-92% 来自抵消结构, refit 可回落
1-2 个量级 (报告 §4.3)。

硬规则: 只允许**全局常数 C#** (5 工况通用); 严禁逐工况常数 K#。
"""

    def _structure_hint(self) -> str:
        try:
            pool, cmat = self.base["pool"], self.base["coef"]
            terms = sorted(zip(pool, cmat[self.j]),
                           key=lambda kv: -abs(float(kv[1])))[:5]
            body = " + ".join(f"({float(v):+.3g})·{k}" for k, v in terms
                              if abs(float(v)) > 1e-4)
            return f"a_{self.j+1}(Θ) ≈ {body}"
        except Exception:                                       # noqa: BLE001
            return "(结构提示不可用)"

    def branch_priors(self, cfg) -> List[str]:
        M = self.j + 1
        return [
            f"F1 饱和族 (假设): a_{M} 随 roloc 饱和 → tanh(C·roloc)、"
            f"C/(C'+roloc)、exp(-C·roloc)。依据: T5/T6/T10 通道旋转响应的"
            f"既有实验一致有效 (报告附录 A 各 a_j 均含此类结构)。",
            f"F2 近壁族 (假设): 壁面衰减/增长 → yw、yw²、log(yw+C)、1/(C+yw)。"
            f"依据: 冠军公式 a2/a3 的 yw 主导项 (报告附录 A)。",
            f"F3 有理族 (假设): Speziale 形状 → C1/(1+C2·eta2)、eta2/(C+eta2)、"
            f"log((eta2+ε)/(eta2+1))。依据: 系数函数的光滑性 + 量级控制。",
            f"F4 对数律族 (假设): √roloc·(C·log(yw)+C') 型可分离乘积。"
            f"依据: V4b 标量冠军公式所在族 (yw→0 发散须配饱和因子)。",
        ]

    def verified_assertions(self) -> List[str]:
        return [
            "b23 与 T1/T5/T6/T10 的 23 分量全为偶对称 (C1 探针核实, even≈+1.00)。",
            f"本域共 J={self.base['J']} 条通道 ({'|'.join(self.chan_name)})。",
            "5 工况必须共享同一组全局常数 (C4 核实)。",
            "Θ 全部为无量纲量 (C6 核实)。",
            "结构化联合解 (48 参数) 的 in-sample 上界 exists — loss 0.003-0.004 "
            "(报告 §3.1 参照系, 不可外推)。",
        ]

    def unverified_assumptions(self) -> List[str]:
        return [
            "高旋转下准二维化 → 系数呈饱和结构 — 物理解释, 非可核实断言。",
            "近壁低 Re_t 阻尼的 Myong-Kasagi f_μ 型形式 — 类比假设。",
            "SR 中间口径与终口径 refit 的差 (84-92%) 全部来自抵消结构 — "
            "机理归因, 非逐点核实。",
        ]

    def skills(self, include=None) -> Dict[str, str]:
        from ...agent.skills import load_skill_dir
        return load_skill_dir(_SKILL_DIR, include=include)

    def seed_skeletons(self, branch: int = 0) -> List[str]:
        """离线/冷启动结构种子 (对应 4 个假设族; DSL 变量 = BASE_VARS)。"""
        fam = [
            # F1 饱和族
            ["C1*(tanh(C2*(roloc)))",
             "C1*(roloc)*(inv(C2*(1)+C3*(roloc)))",
             "C1*(exp(inv(C2*(roloc)+C3*(1))))"],
            # F2 近壁族
            ["C1*(yw)+C2*(yw)^2",
             "C1*(log(C2*(yw)+C3*(1)))",
             "C1*(inv(C2*(yw)+C3*(1)))"],
            # F3 有理族
            ["C1*(eta2)*(inv(C2*(1)+C3*(eta2)))",
             "C1*(zeta2)*(inv(C2*(1)+C3*(zeta2)))",
             "C1*(s2o2)*(inv(C2*(1)+C3*(s2o2)))"],
            # F4 对数律族
            ["C1*(sqrt(roloc))*(C2*(log(yw))+C3*(roloc))",
             "C1*(sqrt(roloc))+C2*(roloc)*(log(C3*(yw)))",
             "C1*(ret)*(inv(C2*(1)+C3*(ret)))"],
        ]
        k = int(branch) % len(fam)
        out = list(fam[k])
        for jj in range(len(fam)):
            if jj != k:
                out.append(fam[jj][0])
        return out

    # ---- 早停 ----
    def target_met(self, cand) -> bool:
        if not cand.is_admissible():
            return False
        loss = float(cand.loss or float("inf"))
        return loss <= self.target_loss


_SKILL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")
