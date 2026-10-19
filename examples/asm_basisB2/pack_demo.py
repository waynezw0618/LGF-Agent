"""从上层工程完整数据集抽取 basisB2 示例用的精简 npz (独立可运行)。

精简原则 (与 asm_b23 的 demo 相同):
- 每 129 点工况抽 64 点 (均匀 Stride, 保 yh 网格结构与偶对称采样);
- 只保留 basisB2 需要的键: valid/yh/scalars/b6/{T1,T5,T6,T10}[...,23],
  T 通道退化为 (N,) 的 23 分量 (load_channels 兼容两种布局);
- 保留 ro/k/eps 元数据键。

用法:
    python examples/asm_basisB2/pack_demo.py                # 从默认位置找
    python examples/asm_basisB2/pack_demo.py --src <dir> --dst <dir>
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PARENT = os.path.dirname(REPO)
IDX23 = 5
ROS = [7.5, 15.0, 30.0, 75.0, 150.0]
CHANS = ["T1", "T5", "T6", "T10"]


def default_src() -> str:
    for cand in (os.path.join(PARENT, "outputs", "asm"),
                 os.path.join(PARENT, "outputs", "asm")):
        if os.path.exists(os.path.join(cand, "dataset_trueasm_e.npz")):
            return cand
    return ""


def pack_one(src_npz: str, dst_npz: str, stride: int = 2) -> int:
    d = np.load(src_npz, allow_pickle=True)
    out = {"_note": f"basisB2 demo; stride={stride}; from {os.path.basename(src_npz)}"}
    n = 0
    for i, ro in enumerate(ROS):
        p = f"c{i}_"
        if (p + "valid") not in d.files:
            continue
        valid = d[p + "valid"].astype(bool)
        idx = np.where(valid)[0][::stride]                  # 保序均匀抽取
        for key in ("yh", "k", "eps"):
            if (p + key) in d.files:
                arr = np.asarray(d[p + key])
                if arr.ndim == 1 and arr.shape[0] == valid.shape[0]:
                    out[p + key] = arr[idx]
        if (p + "ro") in d.files:
            out[p + "ro"] = np.asarray(d[p + "ro"])
        # 抽稀后的数组已全部是有效点 → valid 全 True (长度 = idx.size)
        out[p + "valid"] = np.ones(idx.size, dtype=bool)
        if (p + "scalars") in d.files:
            arr = np.asarray(d[p + "scalars"])
            out[p + "scalars"] = arr[idx] if arr.shape[0] == valid.shape[0] else arr
        if (p + "b6") in d.files:
            arr = np.asarray(d[p + "b6"])
            out[p + "b6"] = arr[idx] if arr.shape[0] == valid.shape[0] else arr
        for t in CHANS:
            if (p + t) in d.files:
                arr = np.asarray(d[p + t])
                if arr.ndim == 2 and arr.shape[0] == valid.shape[0]:
                    out[p + t] = arr[idx, IDX23]            # 23 分量, (N,)
                elif arr.ndim == 1 and arr.shape[0] == valid.shape[0]:
                    out[p + t] = arr[idx]
        n = max(n, int(idx.size))
    os.makedirs(os.path.dirname(os.path.abspath(dst_npz)), exist_ok=True)
    np.savez_compressed(dst_npz, **out)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=default_src(),
                    help="含 dataset_trueasm_{v,g,e}.npz 的目录")
    ap.add_argument("--dst", default=os.path.join(HERE, "data"))
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--variants", default="v,g,e")
    a = ap.parse_args()
    if not a.src:
        raise SystemExit("找不到源目录, 请用 --src 指向上层 outputs/asm")
    for v in a.variants.split(","):
        src = os.path.join(a.src, f"dataset_trueasm_{v}.npz")
        dst = os.path.join(a.dst, f"dataset_trueasm_{v}_demo.npz")
        n = pack_one(src, dst, a.stride)
        sz = os.path.getsize(dst) / 1024
        print(f"[pack] {v}: {n} 点/工况 (stride={a.stride}) -> {dst} ({sz:.1f} KB)")


if __name__ == "__main__":
    main()
