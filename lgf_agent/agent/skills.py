"""lgf_agent.agent.skills — 领域知识卡 (read_skill 的数据源).

知识卡是纯 markdown 文件, 放在域目录的 `skills/` 下, 文件名 (去扩展名) 即主题名。
`DomainAdapter.skills()` 负责决定加载哪些; 本模块只负责读取与合并。

为什么是文件而不是 prompt 硬编码: 方案 §3.6 —— 让"领域知识"与"提示词工程"解耦,
审稿人可以直接读这些卡来判断我们给了模型什么先验 (可审计性)。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence


def load_skill_file(path: str) -> Optional[tuple]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return (os.path.splitext(os.path.basename(path))[0], f.read())
    except OSError:
        return None


def load_skill_dir(d: str, *, include: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """加载目录下所有 .md 为 {topic: text}; include 非空时只保留列出的主题。"""
    out: Dict[str, str] = {}
    if not d or not os.path.isdir(d):
        return out
    want = set(include) if include else None
    for fn in sorted(os.listdir(d)):
        if not fn.lower().endswith((".md", ".txt")):
            continue
        topic = os.path.splitext(fn)[0]
        if want is not None and topic not in want:
            continue
        got = load_skill_file(os.path.join(d, fn))
        if got:
            out[got[0]] = got[1]
    return out


def merge(*dicts: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for d in dicts:
        out.update(d or {})
    return out
