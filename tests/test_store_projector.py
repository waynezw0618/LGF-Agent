"""L2 状态层测试: 候选契约 / append-only 回放 / legacy 回灌 / 投影。"""
from __future__ import annotations

import json
import os

import pytest

from lgf_agent.projector import pareto_front, project, render
from lgf_agent.store import Archive, Candidate


def _c(dsl, loss, nodes):
    return Candidate(dsl=dsl, loss=loss, complexity={"nodes": nodes, "params": 1})


def test_candidate_contract_and_compact():
    c = Candidate(dsl="C1*x", loss=0.12, theta=[1.0],
                  guardrails={"collapse": {"verdict": "pass"},
                              "C9": {"verdict": "violate"}})
    assert not c.is_admissible()
    assert c.violations() == ["C9"]
    s = c.compact()
    assert "C1*x" in s and len(s) <= 400


def test_archive_append_only_replay(tmp_path):
    p = tmp_path / "a.jsonl"
    ar = Archive(str(p), task="t")
    c1 = ar.add(_c("C1*x", 1.0, 3))
    c1.status = "admitted"
    ar.update(c1)                       # 状态推进 → 追加一条新记录
    ar.add(_c("C1*x^2", 0.5, 4))
    ar.record("note", hello=1)

    # 磁盘上是 append-only: 3 条记录, 不是覆写
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 4

    # 重放后状态一致
    ar2 = Archive(str(p), task="t")
    assert len(ar2) == 2
    assert ar2.get(c1.id).status == "admitted"
    assert any(e.get("event") == "note" for e in ar2.events)


def test_archive_ranked_and_best(tmp_path):
    ar = Archive(str(tmp_path / "b.jsonl"))
    ar.add(_c("bad", 5.0, 3))
    good = ar.add(_c("good", 0.1, 5))
    assert ar.best() is not None and ar.best().dsl == "good"
    assert ar.ranked()[0].id == good.id


def test_legacy_replay_layouts(tmp_path):
    d = tmp_path / "runs_x"
    d.mkdir()
    (d / "a_final.json").write_text(json.dumps({
        "best": {"dsl": "C1*x", "loss": 0.3, "theta": [1.0]},
        "top_final": [{"dsl": "C2*x", "loss": 0.4}],
        "profile": {"var_names": ["x", "y"]},
    }), encoding="utf-8")
    (d / "summary_A.json").write_text(json.dumps({
        "targets": {"s12": {"dsl": "C3*x", "loss": 0.9}},
        "var_names": ["x"],
    }), encoding="utf-8")

    ar = Archive.from_legacy_runs(str(tmp_path), archive_path=str(tmp_path / "ar.jsonl"),
                                  globs=["*_final.json", "summary_*.json"])
    assert len(ar) == 3
    dsLs = {c.dsl for c in ar}
    assert dsLs == {"C1*x", "C2*x", "C3*x"}
    got = ar.get(next(c.id for c in ar if c.dsl == "C1*x"))
    assert got.var_names == ["x", "y"]


def test_legacy_replay_tolerates_bad_files(tmp_path):
    d = tmp_path / "runs_y"
    d.mkdir()
    (d / "broken_final.json").write_text("{not json", encoding="utf-8")
    (d / "ok_final.json").write_text(json.dumps({"dsl": "C1*x", "loss": 1.0}),
                                     encoding="utf-8")
    ar = Archive.from_legacy_runs(str(tmp_path))
    assert len(ar) == 1
    assert any(e.get("event") == "legacy_replay_error" for e in ar.events)


# ---------------------------------------------------------------- projector
def test_pareto_front_dedup_and_dominance():
    cs = [_c("A", 1.0, 10), _c("A", 1.0, 10),      # 重复
          _c("B", 0.5, 4),                          # 同时更优 → 支配 A
          _c("C", 2.0, 2)]                          # 更简单但更差 → 在前沿
    front = [c.dsl for c in pareto_front(cs)]
    assert "A" not in front
    assert set(front) == {"B", "C"}


def test_project_views_and_shuffle():
    cs = [_c("A", 1.0, 9), _c("B", 0.2, 3), _c("C", 0.3, 3)]
    assert [c.dsl for c in project(cs, view="single_best", shuffle=False)] == ["B"]
    assert [c.dsl for c in project(cs, view="topk", max_n=2, shuffle=False)] == ["B", "C"]
    # shuffle 只改顺序, 不改入选集合
    assert {c.dsl for c in project(cs, view="topk", max_n=2, shuffle=True)} == {"B", "C"}
    assert render(cs, view="topk", max_n=2).startswith("#")


def test_project_unknown_view_raises():
    with pytest.raises(ValueError):
        project([_c("A", 1.0, 1)], view="nope")
