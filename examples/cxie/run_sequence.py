#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按序调用 `md.py`：分别在 300K、2000K、4000K 下运行（每次同时设置起止温度）。
只修改 `input.toml` 中 `MD` 区块的 `temperature_init` 与 `temperature_final`，其余参数保持不变。

用法：直接运行此脚本即可（会按 `input.toml` 的其它 MD 参数执行）。
"""
import sys
import os
import json
import copy
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'input.toml'
MD_SCRIPT = ROOT / 'md.py'

if not INPUT.exists():
    print('错误：找不到 input.toml')
    sys.exit(1)

try:
    if sys.version_info >= (3, 11):
        import tomllib as toml
        def load_toml(p):
            # tomllib.loads 需要 str，使用 read_text() 返回文本
            return toml.loads(p.read_text(encoding='utf-8'))
    else:
        import toml
        def load_toml(p):
            return toml.load(p.open('r', encoding='utf-8'))
except Exception as e:
    print('错误：无法导入 toml 支持：', e)
    sys.exit(1)


def run_with_temp(base_cfg, temp):
    cfg = copy.deepcopy(base_cfg)
    if 'MD' not in cfg:
        raise SystemExit('input.toml 中缺少 MD 区块')
    cfg['MD']['temperature_init'] = float(temp)
    cfg['MD']['temperature_final'] = float(temp)

    env = os.environ.copy()
    env['MP_CONFIG'] = json.dumps(cfg)

    print(f"\n== 调用 md.py: 温度 {temp} K 开始 ==")
    try:
        subprocess.run([sys.executable, str(MD_SCRIPT)], check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"md.py 返回非零状态: {e.returncode}")
        raise
    # 重命名结果文件以避免后续阶段覆盖
    from shutil import move
    out_final = ROOT / 'final_structure.xyz'
    out_traj = ROOT / 'trajectory.xyz'
    if out_final.exists():
        dst = ROOT / f"final_structure_{int(temp)}K.xyz"
        try:
            move(str(out_final), str(dst))
            print(f"已重命名 final_structure -> {dst.name}")
        except Exception as e:
            print(f"重命名 final_structure 失败: {e}")
    else:
        print('注意：未找到 final_structure.xyz，跳过重命名')

    if out_traj.exists():
        dst = ROOT / f"trajectory_{int(temp)}K.xyz"
        try:
            move(str(out_traj), str(dst))
            print(f"已重命名 trajectory -> {dst.name}")
        except Exception as e:
            print(f"重命名 trajectory 失败: {e}")
    else:
        print('注意：未找到 trajectory.xyz，跳过重命名')


def main():
    base = load_toml(INPUT)

    # 顺序：300K -> 2000K -> 4000K
    sequence = [300.0, 2000.0, 4000.0]
    for t in sequence:
        run_with_temp(base, t)

    print('\n全部阶段完成')


if __name__ == '__main__':
    main()
