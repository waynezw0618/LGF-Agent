"""约束层测试: 数值探针必须给出可复现证据, 且不误报。"""
from __future__ import annotations

import numpy as np
import pytest

from lgf_agent.verify import (PASS, VIOLATE, Constraint, ProbeResult,
                              VerifyContext, Verifier, default_hard_rules)


def _ctx(dsl, var_names, cases, theta=None, **extras):
    return VerifyContext(dsl=dsl, var_names=var_names, cases=cases,
                         theta=theta, extras=extras)


def test_collapse_rule_flags_constant(synthetic_cases):
    cases, vn = synthetic_cases
    ver = Verifier(default_hard_rules())
    r = ver.run_all(_ctx("C1 * (1)", vn, cases, theta=[0.5]))
    assert r["collapse"]["verdict"] == VIOLATE


def test_collapse_rule_accepts_real_function(synthetic_cases):
    cases, vn = synthetic_cases
    ver = Verifier(default_hard_rules())
    r = ver.run_all(_ctx("C1 * (sqrt(x))", vn, cases, theta=[0.5]))
    assert r["collapse"]["verdict"] == PASS


def test_param_blowup(synthetic_cases):
    cases, vn = synthetic_cases
    ver = Verifier(default_hard_rules())
    assert ver.run_all(_ctx("C1 * (sqrt(x))", vn, cases,
                            theta=[1e5]))["param_blowup"]["verdict"] == VIOLATE
    assert ver.run_all(_ctx("C1 * (sqrt(x))", vn, cases,
                            theta=[1.0]))["param_blowup"]["verdict"] == PASS


def test_cross_case_stability(synthetic_cases):
    cases, vn = synthetic_cases
    ver = Verifier(default_hard_rules())
    good = ver.run_all(_ctx("C1 * (sqrt(x)) + C2 * (x)", vn, cases,
                            theta=[0.5, 0.3]))
    assert good["cross_case_stability"]["verdict"] == PASS


def test_cross_case_stability_flags_scale_mismatch():
    """构造"只吃得下 A 工况"的场景: 两工况目标量级差 100×,
    单一全局系数必然让 NMSE 的 max/min 远超阈值。"""
    rng = np.random.default_rng(1)
    xa = rng.uniform(0.2, 1.0, 100)
    xb = rng.uniform(0.2, 1.0, 100)
    cases = [dict(X=xa[:, None], y=100.0 * xa, name="A"),
             dict(X=xb[:, None], y=1.0 * xb, name="B")]
    ver = Verifier(default_hard_rules(cross_case_ratio=5.0))
    r = ver.run_all(_ctx("C1 * (x)", ["x"], cases, theta=[50.0]))
    ev = r["cross_case_stability"]["evidence"]
    assert r["cross_case_stability"]["verdict"] == VIOLATE
    assert ev["ratio"] > 5.0


def test_range_guard_catches_definition_domain_errors(synthetic_cases):
    cases, vn = synthetic_cases
    ver = Verifier(default_hard_rules())
    # 人为构造: theta 使 exp 爆炸
    r = ver.run_all(_ctx("C1 * (exp(C2 * (x)))", vn, cases, theta=[1.0, 900.0]))
    assert r["range_guard"]["verdict"] == VIOLATE


def test_forbidden_ops(synthetic_cases):
    cases, vn = synthetic_cases
    ver = Verifier(default_hard_rules(forbidden=["exp"]))
    assert ver.run_all(_ctx("C1 * (exp(x))", vn, cases,
                            theta=[1.0]))["forbidden_ops"]["verdict"] == VIOLATE
    assert ver.run_all(_ctx("C1 * (sqrt(x))", vn, cases,
                            theta=[1.0]))["forbidden_ops"]["verdict"] == PASS


def test_non_numeric_constraint_degrades_to_unverified(synthetic_cases):
    cases, vn = synthetic_cases

    class Mystic(Constraint):
        name = "mystic"
        numeric = False

        def check(self, ctx):                       # pragma: no cover
            raise AssertionError("不该被调用")

    ver = Verifier([Mystic()])
    r = ver.run_all(_ctx("C1 * (x)", vn, cases))["mystic"]
    assert r["verdict"] == "unverified"
    assert "假设" in r["note"]


def test_probe_exception_becomes_unverified(synthetic_cases):
    cases, vn = synthetic_cases

    class Boom(Constraint):
        name = "boom"

        def check(self, ctx):
            raise RuntimeError("probe crash")

    r = Verifier([Boom()]).run_all(_ctx("C1 * (x)", vn, cases))["boom"]
    assert r["verdict"] == "unverified"


def test_verifier_summary_and_verdict(synthetic_cases):
    cases, vn = synthetic_cases
    rep = Verifier(default_hard_rules()).run_all(
        _ctx("C1 * (1)", vn, cases, theta=[0.5]))
    s = Verifier.summarize(rep)
    assert s["violations"] and s["admissible"] is False
    assert Verifier.verdict(rep) == "reject"

    rep2 = Verifier(default_hard_rules()).run_all(
        _ctx("C1 * (sqrt(x)) + C2 * (x)", vn, cases, theta=[0.5, 0.3]))
    assert Verifier.verdict(rep2) in ("pass", "return_for_refinement")
