"""basisB2 域测试: 多通道系数口径 + 白化 + 联合 refit (数值对齐上层工程).

对账锚点 (报告 BasisB2_full_matrix_report.md §3.1):
- e_lgf_mg_big 终口径 0.0576 (worst 0.068), 中间口径 0.7918;
- 结构化联合解 (48 参数) 上界 ~0.003-0.004 (in-sample)。

需要上层工程完整数据集时自动跳过 (标记 skip); demo 数据 (64 点) 用于
端到端链路, 数值对齐测试用完整数据 (129 点)。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from lgf_agent import LGFConfig
from lgf_agent.domains.asm.basisb2 import (
    BasisB2Domain, BASE_VARS, coef_pool, eval_nmse, load_channels,
    reconstruct, sr_cases_for_target, stage_base, weight_vector)
from lgf_agent.domains.asm.refit import (collect_matrix, compile_expr,
                                         joint_refit, refit_intermediate)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO)
FULL_DIR = os.path.join(PARENT, "outputs", "asm")
RUNS_DIR = os.path.join(FULL_DIR, "runs_basisB2")
DEMO_DIR = os.path.join(REPO, "examples", "asm_basisB2", "data")


def _full_paths():
    paths = {v: os.path.join(FULL_DIR, f"dataset_trueasm_{v}.npz")
             for v in ("v", "g", "e")}
    if all(os.path.exists(p) for p in paths.values()):
        return paths
    return None


def _demo_paths():
    paths = {v: os.path.join(DEMO_DIR, f"dataset_trueasm_{v}_demo.npz")
             for v in ("v", "g", "e")}
    if all(os.path.exists(p) for p in paths.values()):
        return paths
    return None


@pytest.fixture(scope="module")
def full_paths():
    p = _full_paths()
    if p is None:
        pytest.skip("上层工程完整数据集不可用 (outputs/asm/dataset_trueasm_{v,g,e}.npz)")
    return p


@pytest.fixture(scope="module")
def any_paths():
    return _full_paths() or _demo_paths()


# ---------------------------------------------------------------------------
# 1. 数据加载 + stage base (数值对齐上层 rc_asm_basisB2.py)
# ---------------------------------------------------------------------------
def test_load_channels_shapes_and_norm(full_paths):
    cases = load_channels("e", full_paths, cnorm="percase")
    assert len(cases) == 5
    assert [c["ro"] for c in cases] == [7.5, 15.0, 30.0, 75.0, 150.0]
    for c in cases:
        assert c["ch"].shape == (c["Nv"], 4)              # J=4 (T1/T5/T6/T10)
        # percase RMS 归一: 每列 RMS = 1
        rr = np.sqrt(np.mean(c["ch"] ** 2, axis=0))
        np.testing.assert_allclose(rr, 1.0, rtol=1e-10)
        assert set(c["f"].keys()) == set(BASE_VARS)


def test_stage_base_matches_upstream(full_paths):
    """结构化联合解应复现上层 stage base 的 g 基 0.0034 / e 基 0.0037 量级。

    上层 (10-18 跑批): g 0.0034 / e 0.0037 (in-sample, 48 参数)。
    """
    for variant, ref in (("g", 0.0034), ("e", 0.0037)):
        cases = load_channels(variant, full_paths, cnorm="percase")
        base = stage_base(cases)
        assert base["J"] == 4 and base["K"] == 12
        assert base["coef"].shape == (4, 12)
        # in-sample 上界: 与报告参照系同量级 (不逐位对齐, 跑批 RNG 无关 — lstsq 确定)
        assert base["loss_vw"] < 0.01, f"{variant}: {base['loss_vw']}"
        assert len(base["targets"]) == 4
        # β 靶标 O(1) 量级 (白化/归一的承诺)
        for v in base["beta_rms"]:
            assert 1e-3 < v < 10.0


def test_sr_cases_weight_and_scale(full_paths):
    """SR 拟合 cases: y=β_j/s_j 归一, w=Ũ_j²/(Nv·var_c) (上层逐字口径)。"""
    cases = load_channels("g", full_paths, cnorm="percase")
    base = stage_base(cases)
    for j in range(base["J"]):
        cs, scale = sr_cases_for_target(base, j)
        assert scale == pytest.approx(base["beta_rms"][j], rel=1e-12)
        for c, raw in zip(cs, cases):
            assert c["X"].shape == (raw["Nv"], 6)
            # y 归一: pooled RMS over 5 cases = 1
            np.testing.assert_allclose(
                np.sqrt(np.mean(np.concatenate([x["y"] for x in cs]) ** 2)),
                1.0, rtol=1e-10)
            # w 口径逐字对齐
            np.testing.assert_allclose(
                c["w"], raw["U"][:, j] ** 2 / (raw["Nv"] * raw["var_c"]),
                rtol=1e-12)


# ---------------------------------------------------------------------------
# 2. BasisB2Domain: 标准 LGF-Agent 任务
# ---------------------------------------------------------------------------
def test_domain_build_and_card(full_paths):
    cases = load_channels("e", full_paths, cnorm="percase")
    base = stage_base(cases)
    dom = BasisB2Domain(base, target_index=0, variant="e")
    cs, vn = dom.build_cases(LGFConfig())
    assert vn == BASE_VARS
    assert len(cs) == 5
    assert dom.channel_name == "T1s"
    card = dom.task_card()
    for key in ("eta2", "zeta2", "s2o2", "ret", "roloc", "yw",
                "全局常数", "联合 refit"):
        assert key in card
    # extras 结构与 asm 域一致 (CaseExtra 对象, C2 约束要用)
    ex = dom.extras()
    from lgf_agent.domains.asm.dataset import CaseExtra
    assert isinstance(ex["case_extras"][0], CaseExtra)
    assert ex["case_extras"][0].channels.shape == (cs[0]["X"].shape[0], 4)


def test_domain_seed_skeletons_parse(full_paths):
    """4 个假设族的种子全部可按 BASE_VARS 解析 (离线管线的入口)。"""
    from lgf_agent.core import dsl as _dsl
    cases = load_channels("g", full_paths, cnorm="percase")
    base = stage_base(cases)
    dom = BasisB2Domain(base, target_index=1, variant="g")
    for br in range(4):
        for s in dom.seed_skeletons(branch=br):
            _dsl.parse_dsl(s, BASE_VARS)          # 不抛即通过


# ---------------------------------------------------------------------------
# 3. 联合 refit: 独立复算报告冠军 (0.0576)
# ---------------------------------------------------------------------------
def test_refit_replay_champion(full_paths):
    """e_lgf_mg_big: 中间 0.7918 → 终 0.0576, LGF-Agent 包内独立复算 (±5%)。"""
    p = os.path.join(RUNS_DIR, "e_lgf_mg_big", "result.json")
    if not os.path.exists(p):
        pytest.skip("runs_basisB2/e_lgf_mg_big 不可用")
    run = json.load(open(p, encoding="utf-8"))
    rows = run["targets"]
    dsls = [r["dsl"] for r in rows]
    scales = [float(r["scale"]) for r in rows]
    thetas = [r.get("theta") for r in rows]
    cases = load_channels("e", full_paths, cnorm="percase")
    from lgf_agent.domains.asm.basisb2 import whiten
    whiten(cases, use_whiten=False)

    inter = refit_intermediate(cases, dsls, scales, thetas)
    assert inter["loss_vw"] == pytest.approx(run["loss_vw"], rel=0.02)

    refit = joint_refit(cases, dsls, scales, thetas)
    assert refit is not None
    ref = run["refit"]["loss_vw"]
    assert refit["loss_vw"] == pytest.approx(ref, rel=0.05)
    assert refit["worst"] == pytest.approx(run["refit"]["worst"], rel=0.05)
    assert refit["n_params"] == run["refit"]["n_params"]
    # refit 至少不差于中间口径的 1/5 (抵消结构被治愈的报告机理)
    assert refit["loss_vw"] < inter["loss_vw"] / 5.0


def test_compile_expr_pysr_inline_constants():
    """PySR 内联常量 → 参数化 C9xx (与 LGF 同场抛光的协议)。"""
    fn, params, th0 = compile_expr("2.5*yw + 0.3*sin(roloc)")
    assert params and all(p.startswith("C") for p in params)
    assert len(params) == 2 and sorted(th0) == [0.3, 2.5]
    # BASE_VARS 顺序 [eta2,zeta2,s2o2,ret,roloc,yw] → roloc=col4, yw=col5
    n = 5
    X = np.zeros((n, 6))
    X[:, 4] = np.linspace(0.1, 2, n)                  # roloc
    X[:, 5] = np.linspace(0.1, 1, n)                  # yw
    order = {p: v for p, v in zip(params, th0)}
    v = fn(*[X[:, i] for i in range(6)], *[order[p] for p in params])
    exp = 2.5 * X[:, 5] + 0.3 * np.sin(X[:, 4])
    np.testing.assert_allclose(v, exp, rtol=1e-12)


# ---------------------------------------------------------------------------
# 4. 端到端 (demo 数据): stage base → domain → 离线 agent → 两级口径
# ---------------------------------------------------------------------------
def test_end_to_end_offline_demo(any_paths):
    """demo 数据全链路: 精简 npz → base → 单系数任务 → 中间重组 vs refit。"""
    from lgf_agent.agent.scheduler import run_agent
    from lgf_agent.core import dsl as _dsl
    from lgf_agent.core.paramfit import fit_joint

    cases = load_channels("e", any_paths, cnorm="percase")
    base = stage_base(cases)
    dom = BasisB2Domain(base, target_index=0, variant="e")
    cfg = LGFConfig(population=6, generations=2, time_budget_s=60.0,
                    gp_starts=2, gp_fit_budget_s=4.0, final_starts=3,
                    final_maxiter=60, max_edges=12, max_params=4, seed=0,
                    agent_mode=True, agent_R=1, agent_C=1, agent_L=2,
                    agent_K=1, agent_inspector=True)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        res = run_agent(dom, cfg, out_dir=td, task_name="t_basisb2")
    assert res["n_candidates"] > 0
    assert res["best"] is not None and np.isfinite(res["best"]["loss"])

    # 两级口径: 结构种子 4 条 → 中间重组 → refit 变好 (抵消治愈机理)
    dsls, scales, thetas = [], [], []
    for j in range(4):
        dj = BasisB2Domain(base, target_index=j, variant="e")
        dj_cases, _ = dj.build_cases(cfg)
        seed = dj.seed_skeletons(branch=j)[0]
        g = _dsl.parse_dsl(seed, list(dj.var_names))
        fr = fit_joint(g, dj_cases, n_starts=2, maxiter=60,
                       time_budget_s=6.0, seed=0)
        dsls.append(seed); scales.append(dj.scale)
        thetas.append([float(v) for v in fr.theta])
    inter = refit_intermediate(cases, dsls, scales, thetas)
    refit = joint_refit(cases, dsls, scales, thetas)
    assert refit is not None
    assert refit["loss_vw"] < inter["loss_vw"]       # refit 不变差 (LM 从同一点出发)


# ---------------------------------------------------------------------------
# 5. 矩阵汇总 (纯读取)
# ---------------------------------------------------------------------------
def test_collect_matrix_reads_all_groups():
    if not os.path.isdir(RUNS_DIR):
        pytest.skip("runs_basisB2 不可用")
    rows = collect_matrix(RUNS_DIR)
    assert len(rows) >= 27
    by = {r["group"]: r for r in rows}
    champ = by["e_lgf_mg_big"]
    assert champ["loss_refit"] == pytest.approx(0.0576, abs=2e-3)
    assert by["m_lgf_go_big"]["loss_refit"] == pytest.approx(0.0722, abs=5e-3)
    assert by["v_lgf_go"]["loss_refit"] == pytest.approx(2.231, abs=0.02)
