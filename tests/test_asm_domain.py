"""ASM 域适配器测试: 数据加载 / 任务卡 / C1 奇偶性机器判定 / 九条约束。"""
from __future__ import annotations

import numpy as np
import pytest

from lgf_agent import LGFConfig
from lgf_agent.domains.asm import ASMDomain, asm_constraints
from lgf_agent.domains.asm.channels import NONZERO_CHANNELS_23
from lgf_agent.domains.asm.constraints import even_score


@pytest.fixture()
def dom(demo_dataset):
    return ASMDomain.from_dataset(demo_dataset, variant="g")


def test_build_cases(dom):
    cfg = LGFConfig()
    cases, vn = dom.build_cases(cfg)
    assert len(cases) == 5
    assert vn == ["eta2", "zeta2", "s2o2", "ret", "roloc", "yw"]
    for c in cases:
        assert c["X"].shape[1] == len(vn)
        assert c["X"].shape[0] == c["y"].shape[0]
    # 默认口径不设 w (逐工况 NMSE, 与 loss_vw 一致)
    assert "w" not in cases[0]


def test_evaluate_metric_matches_manual(dom):
    cfg = LGFConfig()
    cases, vn = dom.build_cases(cfg)
    preds = [np.zeros_like(c["y"]) for c in cases]     # 常数零预测
    r = dom.evaluate(preds)
    # 常数零预测的逐工况 NMSE ≈ 1 + (mean/std)^2 ≥ 1
    assert r["loss_vw"] >= 1.0
    assert set(r["per_case"]) == {"Ro7.5", "Ro15", "Ro30", "Ro75", "Ro150"}


def test_task_card_and_assertion_split(dom):
    card = dom.task_card()
    assert "b23 = g(Θ) · T23" in card
    assert "全局常数" in card
    # 诚实分开: 可核实断言与不可核实假设必须分列
    assert dom.verified_assertions() and dom.unverified_assumptions()
    assert any("偶对称" in s for s in dom.verified_assertions())


def test_parity_audit_machine_verifies_even_symmetry(dom):
    """P0 判决点第 3 条: 机器判定 T*_23 为偶对称, 且与声明一致。"""
    audit = dom.parity_audit()
    assert audit["consistent"] is True
    for m in NONZERO_CHANNELS_23:
        assert audit["channels"][f"T{m}_23"] > 0.8
    assert audit["channels"]["b23"] > 0.8
    # 对照项: 奇函数应给负分 → 证明探针有区分力
    assert audit["control_odd_yh"] < -0.5


def test_known_bad_assertion_recorded(dom):
    bad = dom.known_bad_assertions()
    assert bad and bad[0]["verdict"] == "FALSE"
    assert "rc_asm_vw_t23_full.py" in bad[0]["source_file"]


def test_even_score_extremes():
    yh = np.linspace(-1, 1, 201)
    even = yh ** 2
    odd = yh
    assert even_score(yh, even) > 0.99
    assert even_score(yh, odd) < -0.99


def test_constraints_cover_c1_to_c9(dom):
    cons = asm_constraints(LGFConfig(), loso_probe=False)
    names = {c.name for c in cons}
    assert len(cons) == 9
    for k in ("C1_parity", "C2_tensor_basis", "C3_range", "C4_global_params",
              "C5_loso", "C6_dimensionless", "C7_numeric", "C8_complexity",
              "C9_synth_cheat"):
        assert k in names


def test_decoy_cases_available_for_c9(dom):
    ex = dom.extras()
    assert ex["case_extras"] and ex["decoy_cases"]
    assert set(ex["decoy_cases"]) >= {"uu", "uv", "uw"}
    # 诱饵与主目标同 X、不同 y
    assert len(ex["decoy_cases"]["uv"]) == 5


def test_seed_skeletons_are_parseable(dom):
    from lgf_agent.core import dsl as _dsl
    vn = dom.var_names
    dsLs = dom.seed_skeletons(0)
    assert len(dsLs) >= 4
    for s in dsLs:
        g = _dsl.parse_dsl(s, vn)
        assert g.complexity()["nodes"] > 0
