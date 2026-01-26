#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轻量工作流主程序（菜单式交互，参考提供框架）。

职责：读取 `input.toml`（按块组织），展示 VASPkit 风格菜单 "MLFF-Zero"，
根据用户选择把配置通过环境变量传给对应子程序并以子进程方式调用。

说明：这是框架骨架，保持简洁，不填充子程序实现细节。
"""
import os
import sys
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))


def load_toml_config():
    """加载 `input.toml`（优先当前工作目录，其次项目根）。"""
    candidates = [Path(os.getcwd()) / "input.toml", PROJECT_ROOT / "input.toml"]
    conf_path = next((p for p in candidates if p.exists()), None)
    if conf_path is None:
        return {}
    try:
        if sys.version_info >= (3, 11):
            import tomllib as _toml
            with conf_path.open('rb') as f:
                data = _toml.load(f)
        else:
            import toml as _toml
            data = _toml.loads(conf_path.read_text())
        return data
    except Exception:
        return {}


def run_subprogram(script_name, extra_env=None):
    """以子进程方式调用项目内脚本，并把 `MP_CONFIG` 作为 JSON 注入环境。"""
    cfg = globals().get('CONFIG', {})
    env = os.environ.copy()
    env['MP_CONFIG'] = json.dumps(cfg)
    if extra_env:
        env.update(extra_env)

    cmd = [sys.executable, str(PROJECT_ROOT / script_name)]
    try:
        subprocess.run(cmd, check=True, env=env)
        return True
    except subprocess.CalledProcessError:
        return False


def menu(options):
    """显示简单菜单并返回被选项的 key。"""
    print('\n' + '=' * 60)
    print('MLFF-Zero — 功能菜单')
    print('=' * 60)
    for i, (key, label) in enumerate(options, 1):
        print(f'{i:2d}. {label}')
    print(' 0. 退出')

    choice = input('\n选择功能编号: ').strip()
    try:
        idx = int(choice)
    except ValueError:
        return None
    if idx == 0:
        return 'exit'
    if 1 <= idx <= len(options):
        return options[idx - 1][0]
    return None


def main():
    global CONFIG
    CONFIG = load_toml_config()
    print('加载配置（部分展示）:')
    print(f'配置已加载（共 {len(CONFIG)} 项）')

    # 菜单映射： key -> (显示名, script)
    menu_items = [
        ('gen', '结构生成', 'structure_generator.py'),
        ('md', '分子动力学', 'md.py'),
    ]

    # 仅将必要映射传给菜单函数（key,label）
    options = [(k, label) for k, label, _ in menu_items]

    while True:
        sel = menu(options)
        if sel in (None, 'exit'):
            print('退出。')
            break

        # 找到对应脚本
        script = next((s for k, _, s in menu_items if k == sel), None)
        if script is None:
            print('无效选择，请重试。')
            continue

        # 将整个配置以 JSON 注入到子进程（子程序可自行解析需要的块）
        print(f'调用子程序: {script} (配置以 MP_CONFIG 注入环境)')
        ok = run_subprogram(script)
        print('完成' if ok else '子程序执行失败')


if __name__ == '__main__':
    main()
#!/usr/bin/env python
