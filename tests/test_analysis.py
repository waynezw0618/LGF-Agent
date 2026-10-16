"""L2 分析层测试: 通道命名解析 + 结构查询 + 数值通道归因 (`lgf_agent.analysis`)。"""
from __future__ import annotations

import numpy as np

from lgf_agent import analysis
from lgf_agent.core import dsl as _dsl
from lgf_agent.core.paramfit import fit_joint
from lgf_agent.store import Archive, Candidate


# ---------------------------------------------------------------------------
# 1. 命名解析 (三套历史命名都必须认出来, 标量变量必须判 None)
# ---------------------------------------------------------------------------
def test_channel_of_var_covers_all_legacy_namings():
    assert analysis.channel_of_var("T1") == "T1"
    assert analysis.channel_of_var("T1_23") == "T1"
    assert analysis.channel_of_var("T1s") == "T1"
    assert analysis.channel_of_var("T5s_v") == "T5(v)"
    assert analysis.channel_of_var("T5_v") == "T5(v)"
    assert analysis.channel_of_var("T10s_e") == "T10(e)"
    assert analysis.channel_of_var("mT1_23") == "T1"
    # 标量 / 壁距 / 局部扩展 一律不是通道
    for name in ("eta2", "zeta2", "s3", "so2", "s2o2", "ret", "roloc", "yw", ""):
        assert analysis.channel_of_var(name) is None, name


def test_channel_groups_and_scalar_only():
    vn = ["roloc", "yw", "T1s", "T5s_v", "T5s_g", "T10s_e"]
    g = analysis.channel_groups(vn)
    assert list(g) == ["T1", "T5(v)", "T5(g)", "T10(e)"]
    assert g["T1"] == [2] and g["T5(v)"] == [3]
    assert analysis.scalar_vars(vn) == ["roloc", "yw"]
    assert analysis.tensor_channels_used(["roloc", "yw"]) == []
    assert analysis.has_tensor_channel(["roloc", "yw"]) is False


# ---------------------------------------------------------------------------
# 2. 数值通道归因 —— 用已知真值的合成算例校准
# ---------------------------------------------------------------------------
def _synth_cases(seed: int = 0, n_cases: int = 3):
    rng = np.random.default_rng(seed)
    cs = []
    for i in range(n_cases):
        n = 80
        roloc = rng.uniform(1.0, 60.0, n)
        t1 = rng.normal(0.0, 1.0, n)
        cs.append({"X": np.column_stack([roloc, t1]),
                   "y": 2.0 * np.sqrt(roloc) + 0.5 * t1,
                   "name": f"synth{i}"})
    return cs


def test_leave_one_channel_out_recovers_known_share():
    """y = 2√roloc + 0.5·T1: 通道 T1 的 contribution 恰为 0.5·T1, rms_share 有解析值。"""
    cases = _synth_cases()
    vn = ["roloc", "T1"]
    g = _dsl.parse_dsl("C1 * sqrt(roloc) + C2 * T1", vn)
    fr = fit_joint(g, cases, n_starts=8, maxiter=200, seed=0)
    rep = analysis.leave_one_channel_out(g, fr.theta, cases, vn)

    assert rep["scalar_only"] is False
    assert rep["loss_full"] < 1e-6                      # 精确可拟合
    rec = rep["channels"]["T1"]
    assert rec["available"] is True
    assert rec["vars"] == ["T1"]

    t1 = np.concatenate([c["X"][:, 1] for c in cases])
    roloc = np.concatenate([c["X"][:, 0] for c in cases])
    pred = 2.0 * np.sqrt(roloc) + 0.5 * t1
    exact_share = float(np.sqrt(np.mean((0.5 * t1) ** 2)) / np.sqrt(np.mean(pred ** 2)))
    assert abs(rec["rms_share"] - exact_share) < 1e-6

    # 去掉 T1 后残差恰为 0.5·T1; loss 口径是 "逐工况 NMSE 再平均", 所以解析值
    # 也必须逐工况算 —— 用 pooled var 会差 0.1% (这正是当初一次性脚本容易踩的坑)
    exact_loss_without = float(np.mean([
        np.mean((0.5 * c["X"][:, 1]) ** 2) / np.var(c["y"]) for c in cases]))
    assert abs(rec["loss_without"] - exact_loss_without) < 1e-9
    assert abs(rec["delta_loss"] - (exact_loss_without - rep["loss_full"])) < 1e-9


def test_leave_one_channel_out_ignores_unused_channel():
    """公式写了通道项但系数被拟合成 0 → rms_share ≈ 0 (或该项被折叠掉)。"""
    cases = _synth_cases(seed=1)
    vn = ["roloc", "T1"]
    g = _dsl.parse_dsl("C1 * sqrt(roloc) + 0 * T1", vn)
    theta = np.asarray(fit_joint(g, cases, n_starts=4, maxiter=100, seed=0).theta)
    rep = analysis.leave_one_channel_out(g, theta, cases, vn)
    rec = rep["channels"].get("T1")
    assert rec is None or rec["rms_share"] < 1e-8


def test_leave_one_channel_out_scalar_only_is_flagged():
    cases = _synth_cases(seed=2)
    vn = ["roloc", "T1"]
    g = _dsl.parse_dsl("C1 * sqrt(roloc)", vn)
    theta = np.asarray(fit_joint(g, cases, n_starts=4, maxiter=200, seed=0).theta)
    rep = analysis.leave_one_channel_out(g, theta, cases, vn)
    # 变量池有 T1, 但公式没用 → LOCO 判为 zero-channel, 并如实记下池里有
    assert rep["scalar_only"] is True
    assert rep["channels"] == {}
    assert rep["pool_channels"] == ["T1"]
    assert rep["unused_pool_channels"] == ["T1"]
    # 变量池本身没有通道时同样成立
    rep2 = analysis.leave_one_channel_out(g, theta, cases, ["roloc"])
    assert rep2["scalar_only"] is True
    assert rep2["pool_channels"] == []


def test_vars_used_in_graph_distinguishes_pool_from_formula():
    vn = ["roloc", "yw", "T1s"]
    g = _dsl.parse_dsl("C1 * sqrt(roloc)", vn)
    assert analysis.vars_used_in_graph(g, vn) == ["roloc"]
    assert analysis.tensor_channels_used(vn) == ["T1"]     # 池里有
    u = analysis.candidate_channel_usage("C1 * sqrt(roloc)", vn)
    assert u["parse_ok"] is True
    assert u["formula_channels"] == [] and u["pool_channels"] == ["T1"]
    assert u["uses_tensor_channel"] is False


# ---------------------------------------------------------------------------
# 3. archive 级结构查询 (替代历史上的 _tmp_t23_contrib.py)
# ---------------------------------------------------------------------------
def test_archive_channel_usage(tmp_path):
    ar = Archive(str(tmp_path / "a.jsonl"), task="t")
    ar.add(Candidate(dsl="C1*sqrt(roloc)", var_names=["roloc", "yw", "T1s"],
                     loss=0.05, status="fitted"))
    ar.add(Candidate(dsl="C1*T1s + C2*T5s_v", var_names=["roloc", "T1s", "T5s_v"],
                     loss=0.07, status="fitted"))
    ar.add(Candidate(dsl="C1*sqrt(roloc)+C2*yw", var_names=["roloc", "yw"],
                     loss=0.06, status="fitted"))

    u = ar.channel_usage()
    assert u["n_candidates"] == 3
    assert u["n_scalar_only"] == 2                    # 公式口径: 第 1、3 条是纯标量
    assert u["n_with_tensor_channel"] == 1
    assert u["n_pool_has_channel"] == 2               # 池口径: 第 1、2 条池里有 T1
    assert u["n_parse_failed"] == 0
    assert u["channels"]["T1"]["n_candidates"] == 1   # 只有第 2 条公式用了 T1
    assert u["channels"]["T1"]["n_pool_candidates"] == 2
    assert u["channels"]["T5(v)"]["n_candidates"] == 1
    # 纯标量组里最好的是 loss=0.05 那条
    assert u["best_scalar_only"][0]["loss"] == 0.05
    # 查询应可从落盘文件重放后复现
    assert Archive(str(tmp_path / "a.jsonl"), task="t").channel_usage()["n_candidates"] == 3


def test_channel_attr_tool_is_registered(synthetic_cases):
    """L1 动作层应暴露 channel_attr, 且在零通道变量池上走结构分支。"""
    from lgf_agent import LGFConfig
    from lgf_agent.tools import ToolContext, default_tools

    ctx = ToolContext(cases=synthetic_cases, var_names=["x"], cfg=LGFConfig())
    reg = default_tools(ctx)
    assert "channel_attr" in reg.names()
    res = reg.dispatch("channel_attr", dsl="C1 * x")
    assert res.ok
    assert res.payload["scalar_only"] is True
