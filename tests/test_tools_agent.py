"""工具层 + 智能体循环测试 (离线通道, 不调任何外部 LLM)。"""
from __future__ import annotations

import numpy as np
import pytest

from lgf_agent import LGFConfig
from lgf_agent.agent.runtime import Audit, CostLedger, parse_agent_response
from lgf_agent.agent.scheduler import build_verifier, run_agent
from lgf_agent.domains.base import DatasetDomain
from lgf_agent.store import Archive
from lgf_agent.tools import ToolContext, ToolResult, default_tools
from lgf_agent.verify import default_hard_rules


@pytest.fixture()
def registry(synthetic_cases, tmp_path):
    cases, vn = synthetic_cases
    cfg = LGFConfig(gp_starts=2, gp_fit_budget_s=3.0, time_budget_s=30.0,
                    max_edges=10, max_params=4, seed=0)
    ctx = ToolContext(cases=cases, var_names=vn, cfg=cfg,
                      archive=Archive(str(tmp_path / "a.jsonl")),
                      verifier=build_verifier(None, cfg),
                      rng=np.random.default_rng(0))
    return default_tools(ctx), ctx


def test_tool_registry_exposes_expected_tools(registry):
    reg, _ = registry
    for name in ("profile_data", "prior_check", "fit_constants", "evaluate_dsl",
                 "attribute", "residual_view", "guardrail_check", "symbol_equiv",
                 "gp_search", "submit_formula", "read_skill"):
        assert reg.get(name) is not None, name
    # 默认不含 code_exec
    assert reg.get("code_exec") is None
    assert "fit_constants" in reg.describe()


def test_dispatch_unknown_tool_is_soft_error(registry):
    reg, _ = registry
    r = reg.dispatch("nope")
    assert r.ok is False and "unknown_tool" in r.evidence.get("error", "")


def test_dispatch_bad_args_is_soft_error(registry):
    reg, _ = registry
    r = reg.dispatch("fit_constants", bogus=1)
    assert r.ok is False


def test_fit_and_guardrail_pipeline(registry):
    reg, ctx = registry
    r = reg.dispatch("fit_constants", dsl="C1 * (sqrt(x)) + C2 * (1)")
    assert r.ok and r.payload["loss"] < 1.0
    g = reg.dispatch("guardrail_check", dsl="C1 * (sqrt(x)) + C2 * (1)",
                     theta=[float(v) for v in r.payload["theta"]])
    assert g.ok and g.payload["verdict"] in ("pass", "return_for_refinement")


def test_residual_view_returns_expression_view(registry):
    reg, _ = registry
    r = reg.dispatch("residual_view", dsl="C1 * (x)", bins=4)
    assert r.ok
    assert r.payload["resid_cases"] and len(r.payload["resid_cases"]) == 5
    assert "bin_rms" in r.evidence["per_case"]["Ro7.5"]


def test_observation_is_bounded(registry):
    reg, _ = registry
    r = reg.dispatch("profile_data")
    assert r.ok and len(r.observation) <= 400


def test_cost_ledger_counts_calls(registry):
    reg, ctx = registry
    reg.dispatch("profile_data")
    reg.dispatch("fit_constants", dsl="C1 * (x)")
    assert ctx.ledger["calls"] >= 2
    assert ctx.ledger["seconds"] > 0


def test_code_exec_disabled_by_default(registry):
    reg, _ = registry
    assert reg.get("code_exec") is None


def test_parse_agent_response_robustness():
    txt = ('```json\n{"thought": "t", "actions": [{"tool": "profile_data"}],'
           ' "candidates": [{"dsl": "C1*x"}]}\n```')
    p = parse_agent_response(txt, tool_names=["profile_data"])
    assert p["actions"][0]["tool"] == "profile_data"
    assert p["candidates"][0]["dsl"] == "C1*x"
    # 非法 JSON 不抛异常
    assert parse_agent_response("not json")["parse_error"] == "no_json_object"
    # 未知工具被标记
    p2 = parse_agent_response('{"actions":[{"tool":"zzz"}]}', tool_names=["a"])
    assert p2.get("unknown_tools") == ["zzz"]


def test_audit_append_and_resume(tmp_path):
    p = tmp_path / "traj.jsonl"
    a = Audit(str(p))
    a.log(event="step", r=0, c=0, ell=3)
    a.log(event="submit", candidate={"id": "c1"})
    a2 = Audit.replay(str(p))
    rp = a2.resume_point()
    assert rp["last_step"]["ell"] == 3
    assert rp["submitted"] == ["c1"]


def test_cost_ledger_per_role():
    l = CostLedger()
    l.add("creator", tokens=100)
    l.add("inspector", tokens=50)
    d = l.to_dict()
    assert d["tokens"] == 150 and d["calls"] == 2
    assert d["per_role"]["creator"]["tokens"] == 100


def test_run_agent_offline_end_to_end(synthetic_cases, tmp_path):
    cases, vn = synthetic_cases
    dom = DatasetDomain(
        cases, vn, name="synth",
        constraints_fn=lambda cfg: default_hard_rules(),
        branch_priors_fn=lambda cfg: ["F1 线性族 (假设)", "F2 幂律族 (假设)"],
        target_met_fn=lambda cand: (cand.loss or 9e9) < 0.02,
    )
    cfg = LGFConfig(population=8, generations=2, time_budget_s=60.0,
                    gp_starts=2, gp_fit_budget_s=2.0, final_starts=3,
                    final_maxiter=60, max_edges=8, max_params=3, seed=0,
                    agent_R=1, agent_C=2, agent_L=2, agent_K=1)
    res = run_agent(dom, cfg, out_dir=str(tmp_path), task_name="synth_smoke")
    assert res["engine"] == "agent"
    assert res["n_candidates"] > 0
    assert res["best"] is not None
    assert res["config"]["C"] == 2 and len(res["config"]["branch_priors"]) == 2
    import os
    assert os.path.exists(res["archive"]) and os.path.exists(res["traj"])
    # 预算字段存在
    assert "cost" in res and res["cost"]["calls"] >= 0


def test_run_agent_respects_time_budget(synthetic_cases, tmp_path):
    cases, vn = synthetic_cases
    dom = DatasetDomain(cases, vn, name="synth2",
                        constraints_fn=lambda cfg: default_hard_rules())
    cfg = LGFConfig(population=4, generations=1, time_budget_s=0.5,
                    gp_starts=1, gp_fit_budget_s=0.5, final_starts=1,
                    final_maxiter=10, max_edges=6, max_params=2, seed=0,
                    agent_L=50)
    res = run_agent(dom, cfg, out_dir=str(tmp_path), task_name="budget")
    assert res["stop_reason"] in ("time_budget", "loop_exhausted")


# ===========================================================================
# usage 真实记账 (P1 前置): last_usage 协议 → CostLedger 优先吃服务端值
# ===========================================================================
class _UsageFakeLLM:
    """模拟回传服务端 usage 的客户端 (OpenAICompatLLM 行为的最小替身)。"""

    def __init__(self, usage=None):
        self.last_usage = usage or {"prompt_tokens": 100,
                                    "completion_tokens": 50,
                                    "total_tokens": 150,
                                    "estimated": False}
        self.calls = 0

    def complete(self, prompt, *, n=1, params=None):
        self.calls += 1
        return ["{\"thought\":\"t\",\"actions\":[],\"candidates\":[]}"]

    def is_heuristic(self):
        return False


class _NoUsageLLM(_UsageFakeLLM):
    """不报告 usage 的客户端 → 应回退字符估算 (estimated 语义)。"""

    def complete(self, prompt, *, n=1, params=None):
        self.calls += 1
        return ["ok"]

    @property
    def usage(self):                     # 显式返回 None (覆盖 BaseLLM 协议)
        return None


def test_creator_ledger_prefers_server_usage(synthetic_cases):
    """Creator 记账: 服务端 usage=150 → ledger 恰好 +150 (不是字符估算)。"""
    from lgf_agent import LGFConfig
    from lgf_agent.agent.runtime import CostLedger
    from lgf_agent.agent.roles import Creator
    from lgf_agent.tools import ToolContext, default_tools

    cases, vn = synthetic_cases
    cfg = LGFConfig()
    ctx = ToolContext(cases=cases, var_names=vn, cfg=cfg)
    reg = default_tools(ctx)
    ledger = CostLedger()
    llm = _UsageFakeLLM()
    cr = Creator(ctx, reg, llm, ledger=ledger)
    cr.propose(pool=[], step=0, branch=0)
    assert llm.calls >= 1
    assert ledger.tokens == 150 * llm.calls          # 精确等于服务端值
    assert ledger.seconds > 0


def test_usage_estimated_fallback(synthetic_cases):
    """无 usage 客户端 → 回退 estimate_tokens (≥1), 不抛错。"""
    from lgf_agent import LGFConfig
    from lgf_agent.agent.runtime import CostLedger, estimate_tokens
    from lgf_agent.agent.roles import Creator
    from lgf_agent.tools import ToolContext, default_tools

    cases, vn = synthetic_cases
    ctx = ToolContext(cases=cases, var_names=vn, cfg=LGFConfig())
    reg = default_tools(ctx)
    ledger = CostLedger()
    llm = _NoUsageLLM()
    Creator(ctx, reg, llm, ledger=ledger).propose(pool=[], step=0, branch=0)
    assert ledger.tokens >= estimate_tokens("ok") * llm.calls
