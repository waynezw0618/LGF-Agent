"""P1 第一步: 实测 GLM-5.3 单题成本 → 推 R-C-L-K 组合 (方案 §5.1).

SRHarness 参照 (LLM-SRBench 单题, DeepSeek-v4-flash): 14.9 min / 1.354M tokens /
$0.0353。我们的 glm-5.3 单价与推理开销不同, 必须实测后才能定 R/C/L/K ——
这就是本脚本的全部职责。

测什么 (basisB2 域 e 基 a2 通道, 真实任务卡):
1. 一次 Creator.propose (完整 prompt: 任务卡+13 工具 schema+分支先验) 的
   tokens/秒/解析率;
2. 一次 Inspector.review 的 tokens/秒;
3. 据此外推: R1C1L30K1 (agent 默认) 一臂的总预算, 与
   agent_token_budget / agent_cost_budget_usd / time_budget_s 三元上限对照;
4. 单价表 (元/1k tokens, 用户可 --price 覆盖) 估算 USD。

用法:
    python examples/asm_basisB2/probe_p1_cost.py                     # 读 .env
    python examples/asm_basisB2/probe_p1_cost.py --n 3 --variant g
输出: run/basisb2_p1_cost/cost_probe.json + 控制台表。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import resolve_datasets, default_out                 # noqa: E402
from lgf_agent import LGFConfig                                   # noqa: E402
from lgf_agent.domains.asm.basisb2 import (                       # noqa: E402
    BasisB2Domain, load_channels, stage_base)

# 智谱 glm-5.3 公开单价 (元/1M tokens, 2026-10 快照; 可 --price-in/--price-out 覆盖)
PRICE_IN_CNY_PER_M = 2.0
PRICE_OUT_CNY_PER_M = 8.0
CNY_PER_USD = 7.2


def load_env_file(path: str) -> bool:
    """读仓库根 .env (gitignored) 注入环境; 返回是否找到 key。"""
    if not os.path.exists(path):
        return False
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    return bool(os.environ.get("LGF_LLM_API_KEY"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--variant", default="e", choices=["v", "g", "e"])
    ap.add_argument("--target", type=int, default=2,
                    help="系数通道 a_j (默认 a2 = T5s 旋转响应, 最有代表性)")
    ap.add_argument("--n", type=int, default=2, help="每个探针重复次数")
    ap.add_argument("--price-in", type=float, default=PRICE_IN_CNY_PER_M)
    ap.add_argument("--price-out", type=float, default=PRICE_OUT_CNY_PER_M)
    ap.add_argument("--out", default=default_out("basisb2_p1_cost"))
    a = ap.parse_args()

    have_key = load_env_file(os.path.join(REPO, ".env")) or \
        bool(os.environ.get("LGF_LLM_API_KEY"))
    if not have_key:
        print("[probe] 未找到 LGF_LLM_API_KEY (仓库 .env 或环境变量)。")
        print("         协议已就绪: 配置后重跑本脚本即可实测。")
        return

    from lgf_agent.core.llm_ops import make_client
    from lgf_agent.agent.roles import Creator, Inspector
    from lgf_agent.agent.runtime import CostLedger
    from lgf_agent.tools import ToolContext, default_tools
    from lgf_agent.agent.scheduler import build_verifier

    url = os.environ.get("LGF_LLM_URL", "https://open.bigmodel.cn/api/paas/v4")
    model = os.environ.get("LGF_LLM_MODEL", "glm-5.3")
    # strict=True: 429/断网时快速失败 —— 测量脚本绝不能拿回退 "{}" 当真数据
    client = make_client("openai", url=url, model=model,
                         api_key=os.environ["LGF_LLM_API_KEY"],
                         max_tokens=16384, timeout=300.0, strict=True,
                         extra_body={"thinking": {"type": "enabled"},
                                     "reasoning_effort": "low"})
    print(f"[probe] backend = {model} @ {url}")

    # 连通性预检 (单独捕获, 给出限额重置等可操作信息)
    try:
        client.complete("回复一个 JSON: {\"ok\": true}", n=1)
    except Exception as e:                                      # noqa: BLE001
        msg = str(e)
        print(f"[probe] 连通性预检失败: {msg[:300]}")
        if "429" in msg or "rate" in msg.lower() or "1302" in msg:
            print("         → 限流/限额: 智谱按 5 小时窗口计费, 等窗口重置后重跑本脚本;")
            print("           或换 reasoning_effort / 降低 --n。")
        return

    # ---- 域 (真实任务卡, 完整 prompt 体量) ----
    paths = resolve_datasets(a.data)
    cases = load_channels(a.variant, paths, cnorm="percase")
    base = stage_base(cases)
    dom = BasisB2Domain(base, target_index=a.target - 1, variant=a.variant)
    cfg = LGFConfig(max_edges=24, max_params=12)
    scases, vn = dom.build_cases(cfg)
    ctx = ToolContext(cases=scases, var_names=vn, cfg=cfg)
    reg = default_tools(ctx)
    verifier = build_verifier(dom, cfg)
    print(f"[probe] task = a_{a.target} ({dom.channel_name}), "
          f"prompt 体量见下")

    # ---- 探针 1: Creator.propose ----
    cr = Creator(ctx, reg, client, ledger=CostLedger(),
                 domain=dom, cfg=cfg)
    prompt = cr.build_prompt(pool=[], step=0, branch=0,
                             branch_prior=(dom.branch_priors(cfg) or [""])[0])
    print(f"[probe] Creator prompt = {len(prompt)} chars "
          f"(≈{int(len(prompt)/2.2)} tokens 估)")
    prop = []
    for i in range(a.n):
        t0 = time.time()
        r = cr.propose(pool=[], step=0, branch=0, k=1)
        u = getattr(client, "last_usage", {}) or {}
        prop.append(dict(seconds=round(time.time() - t0, 2),
                         prompt_tokens=u.get("prompt_tokens"),
                         completion_tokens=u.get("completion_tokens"),
                         reasoning=(u.get("reasoning_tokens", 0) or 0),
                         total=u.get("total_tokens"),
                         n_candidates=len(r.get("candidates") or []),
                         n_actions=len(r.get("actions")
                                       or r.get("observations") or [])))
        print(f"  propose#{i+1}: {prop[-1]}")

    # ---- 探针 2: Inspector.review (用一个真候选) ----
    insp = Inspector(ctx, verifier, client, ledger=CostLedger(), cfg=cfg)
    seed = dom.seed_skeletons(branch=0)[0]
    cand = cr.materialize(seed, rationale="probe")
    rev = []
    for i in range(a.n):
        t0 = time.time()
        out = insp.review(cand)
        u = getattr(client, "last_usage", {}) or {}
        rev.append(dict(seconds=round(time.time() - t0, 2),
                        prompt_tokens=u.get("prompt_tokens"),
                        completion_tokens=u.get("completion_tokens"),
                        reasoning=(u.get("reasoning_tokens", 0) or 0),
                        total=u.get("total_tokens"),
                        verdict=(out or {}).get("verdict")))
        print(f"  review#{i+1}: {rev[-1]}")

    # ---- 外推 R-C-L-K ----
    avg = lambda xs: sum(x for x in xs if x) / max(1, len([x for x in xs if x]))
    tok_c = avg([p["total"] for p in prop])
    tok_i = avg([p["total"] for p in rev])
    sec_c = avg([p["seconds"] for p in prop])
    sec_i = avg([p["seconds"] for p in rev])
    # 一步 = 1 propose (+k=1) + 每候选 1 review; L 步近似 ×L (池视图增长忽略)
    per_step = tok_c + tok_i
    per_step_s = sec_c + sec_i
    arms = {
        "R1C1L30K1(默认)": per_step * 30,
        "R1C1L15K1": per_step * 15,
        "R1C2L15K1": per_step * 30,
        "R2C1L15K1": per_step * 30,
        "R1C1L15K2": per_step * 30,
    }
    in_tok = avg([p["prompt_tokens"] for p in prop])
    out_tok = avg([p["completion_tokens"] for p in prop])
    usd_per_1k = (a.price_in / 1e6 / CNY_PER_USD * 1000,
                  a.price_out / 1e6 / CNY_PER_USD * 1000)

    print("\n=== 单步成本 (propose+review) ===")
    print(f"  tokens/步 ≈ {per_step:.0f} (in≈{in_tok:.0f} out≈{out_tok:.0f})"
          f"  秒/步 ≈ {per_step_s:.1f}")
    print(f"  单价: in ¥{a.price_in}/M, out ¥{a.price_out}/M "
          f"(≈${usd_per_1k[0]:.5}/{usd_per_1k[1]:.5} per 1k)")
    print("\n=== R-C-L-K 外推 (tokens/臂, 时间按串行估) ===")
    for k, v in arms.items():
        usd = v * (in_tok / max(per_step, 1)) * usd_per_1k[0] / 1000 + \
              v * (out_tok / max(per_step, 1)) * usd_per_1k[1] / 1000
        print(f"  {k:<18} {v/1e6:6.2f}M tokens  ≈${usd:5.2f}  "
              f"≈{v / per_step * per_step_s / 60:5.1f} min")

    os.makedirs(a.out, exist_ok=True)
    out = dict(model=model, variant=a.variant, target=a.target,
               propose=prop, review=rev, per_step_tokens=per_step,
               per_step_seconds=per_step_s,
               price=dict(in_cny_per_m=a.price_in, out_cny_per_m=a.price_out),
               arms=arms, prompt_chars=len(prompt))
    with open(os.path.join(a.out, "cost_probe.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[done] -> {os.path.join(a.out, 'cost_probe.json')}")


if __name__ == "__main__":
    main()
