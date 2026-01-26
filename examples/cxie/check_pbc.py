#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查 model.xyz 中各结构的周期性边界条件（pbc）。

用法：
  python check_pbc.py [path_to_xyz]
默认文件：model.xyz。
输出：逐帧报告 pbc（True/False）以及是否三方向均为周期。
"""
import sys
from pathlib import Path

try:
    from ase.io import read
except Exception as exc:
    print(f"错误：无法导入 ASE，请确认已安装 ase。原始错误：{exc}")
    sys.exit(1)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("model.xyz")
    if not path.exists():
        print(f"错误：文件不存在: {path}")
        sys.exit(1)

    try:
        frames = read(path, index=":")
    except Exception as exc:
        print(f"错误：读取 {path} 失败: {exc}")
        sys.exit(1)

    if not frames:
        print(f"文件 {path} 中没有读取到结构帧。")
        return

    print(f"文件: {path} | 帧数: {len(frames)}")
    for i, atoms in enumerate(frames, start=1):
        pbc = atoms.get_pbc()
        # pbc 是一个 (3,) bool 数组
        all_true = bool(pbc.all())
        print(f"帧 {i}: pbc = {pbc.tolist()} | 全周期: {all_true}")


if __name__ == "__main__":
    main()
