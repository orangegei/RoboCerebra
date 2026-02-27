# $ cd /home/shenhaotian/grad/RoboCerebra
# $ python grad/inspect_demo_hdf5.py RoboCerebra_Bench/Ideal/case1/demo.hdf5
"""
=== Top-level keys ===
['data']

=== All nodes (showing up to 200/4) ===
[GROUP]   /data
[GROUP]   /data/demo_1
[DATASET] /data/demo_1/actions  shape=(2554, 7)  dtype=float64  chunks=None  compression=None
[DATASET] /data/demo_1/states  shape=(2554, 71)  dtype=float64  chunks=None  compression=None
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union

import h5py


@dataclass
class H5Node:
    path: str
    type: str  # "group" or "dataset"
    shape: Optional[tuple] = None
    dtype: Optional[str] = None
    chunks: Optional[tuple] = None
    compression: Optional[Union[str, bool]] = None
    attrs: Optional[Dict[str, Any]] = None


def _safe_decode_attr(v: Any) -> Any:
    """将 h5py 的 attrs 值转换成更好打印/序列化的 Python 类型。"""
    # bytes -> str
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode("utf-8")
        except Exception:
            return v
    # numpy scalar/array 在大多数情况下能直接转为 python 标量/list
    try:
        import numpy as np  # noqa: F401
        if hasattr(v, "tolist"):
            return v.tolist()
    except Exception:
        pass
    return v


def inspect_hdf5_fields(h5_path: str, *, include_attrs: bool = True) -> Dict[str, Any]:
    """
    目的：
        列出 demo.hdf5 内部字段结构（不跑仿真），确认是否包含逐帧图像/动作/奖励/done等信号。

    输入：
        h5_path: demo.hdf5 文件路径
        include_attrs: 是否读取每个节点的 attrs（有时 attrs 很多，会慢一点）

    输出（dict）：
        {
          "top_level_keys": [ ... ],
          "nodes": [
              {"path": "...", "type": "group"},
              {"path": "...", "type": "dataset", "shape": ..., "dtype": ..., ...},
              ...
          ]
        }
    """
    nodes: List[H5Node] = []

    with h5py.File(h5_path, "r") as f:
        top_level_keys = list(f.keys())

        def visitor(name: str, obj: Union[h5py.Dataset, h5py.Group]) -> None:
            path = "/" + name if not name.startswith("/") else name

            if isinstance(obj, h5py.Group):
                node = H5Node(
                    path=path,
                    type="group",
                    attrs={k: _safe_decode_attr(v) for k, v in obj.attrs.items()} if include_attrs else None,
                )
                nodes.append(node)
            elif isinstance(obj, h5py.Dataset):
                node = H5Node(
                    path=path,
                    type="dataset",
                    shape=tuple(obj.shape) if obj.shape is not None else None,
                    dtype=str(obj.dtype) if obj.dtype is not None else None,
                    chunks=tuple(obj.chunks) if obj.chunks is not None else None,
                    compression=obj.compression,
                    attrs={k: _safe_decode_attr(v) for k, v in obj.attrs.items()} if include_attrs else None,
                )
                nodes.append(node)
            else:
                # 一般不会走到这
                nodes.append(H5Node(path=path, type=str(type(obj))))

        f.visititems(visitor)

    return {
        "top_level_keys": top_level_keys,
        "nodes": [asdict(n) for n in nodes],
    }


def pretty_print_summary(info: Dict[str, Any], *, max_nodes: int = 200) -> None:
    """把 inspect_hdf5_fields 的结果打印得更直观。"""
    print("=== Top-level keys ===")
    print(info["top_level_keys"])
    print()

    nodes = info["nodes"]
    print(f"=== All nodes (showing up to {max_nodes}/{len(nodes)}) ===")
    shown = 0
    for n in nodes:
        if shown >= max_nodes:
            break
        if n["type"] == "group":
            print(f"[GROUP]   {n['path']}")
        elif n["type"] == "dataset":
            print(
                f"[DATASET] {n['path']}  shape={n.get('shape')}  dtype={n.get('dtype')}  "
                f"chunks={n.get('chunks')}  compression={n.get('compression')}"
            )
        else:
            print(f"[OTHER]   {n['path']}  type={n['type']}")
        shown += 1


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Inspect keys/structure of a demo.hdf5 file.")
    parser.add_argument("h5_path", type=str, help="Path to demo.hdf5")
    parser.add_argument("--no-attrs", action="store_true", help="Do not read HDF5 attrs")
    parser.add_argument("--json", action="store_true", help="Print full result as JSON")
    args = parser.parse_args()

    info = inspect_hdf5_fields(args.h5_path, include_attrs=not args.no_attrs)

    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        pretty_print_summary(info)