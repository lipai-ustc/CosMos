#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分子动力学子程序（由 `main.py` 以子进程方式调用）

从环境变量 `MP_CONFIG` 中读取配置（JSON），根据 `MD` 区块运行 MD。
实现参考 `test.py` 的结构与流程，但参数化为可通过 `input.toml` 控制。
"""
import os
import json
import sys
import time
import numpy as np

try:
    from ase import Atoms, units
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md import Langevin, nptberendsen
    from ase.io import write, read
except Exception as e:
    print(f"缺少依赖或导入失败: {e}")
    sys.exit(1)

# 尝试导入 NequIP 计算器（可在没有模型时回退）
try:
    from nequip.ase import NequIPCalculator
except Exception:
    NequIPCalculator = None

ALLOWED_ELEMENTS = ['Al', 'Mg', 'Ti', 'V', 'Cr', 'Fe', 'Co', 'Ni', 'Cu', 'Zr']


def load_config_from_env():
    raw = os.environ.get('MP_CONFIG', '{}')
    try:
        cfg = json.loads(raw)
    except Exception:
        cfg = {}
    return cfg


def load_allegro_model(model_path, device='cpu'):
    if NequIPCalculator is None:
        print('警告：nequip 未安装，无法加载模型。将使用空计算器占位（仅示例）。')
        return None
    try:
        calc = NequIPCalculator.from_compiled_model(
            compile_path=model_path,
            device=device,
            chemical_species_to_atom_type_map=True
        )
        print(f'成功加载模型：{model_path}')
        return calc
    except Exception as e:
        print(f'加载模型失败：{e}')
        return None


def create_test_structure():
    symbols = ALLOWED_ELEMENTS[:4]
    positions = [
        [0.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [0.0, 2.5, 0.0],
        [0.0, 0.0, 2.5],
    ]
    cell = np.array([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]])
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    return atoms


def linear_temperature(step, total_steps, T_init, T_final):
    """线性温度变化函数"""
    return T_init + (T_final - T_init) * step / total_steps


def run_md_simulation(atoms, calculator, steps, timestep_fs, ensemble,
                      temperature_init, temperature_final, pressure,
                      traj_interval, log_interval, metadata_base=None):
    atoms.calc = calculator
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_init)

    # 判断是否需要温度梯度变化
    use_temp_gradient = (temperature_init != temperature_final)

    # 选择系综和恒温器
    if ensemble == "NVT":
        dyn = Langevin(
            atoms=atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_init,
            friction=0.02
        )
    elif ensemble == "NPT":
        dyn = nptberendsen.NPTBerendsen(
            atoms=atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_init,
            pressure_au=pressure * units.GPa,
            taut=100 * units.fs,  # 温度弛豫时间
            taup=1000 * units.fs,  # 压力弛豫时间
            compressibility=1e-6
        )
    else:
        raise ValueError(f"不支持的系综类型: {ensemble}")

    # 设置温度梯度更新
    if use_temp_gradient:
        def update_temperature():
            """动态更新温度"""
            current_step = dyn.nsteps
            T_current = linear_temperature(
                current_step, steps, temperature_init, temperature_final
            )
            dyn.set_temperature(temperature_K=T_current)

        dyn.attach(update_temperature, interval=1)
        print(f"启用温度梯度变化: {temperature_init}K → {temperature_final}K")

    trajectory = []

    # 更新后的能量打印函数（按 log_interval 打印，不显示实际温度）
    def print_energy():
        n = dyn.nsteps
        try:
            epot = atoms.get_potential_energy()
        except Exception:
            epot = float('nan')

        ekin = atoms.get_kinetic_energy()

        # 计算当前目标温度（仅作为显示目标温度，不打印实际温度）
        T_target = (
            linear_temperature(n, steps, temperature_init, temperature_final)
            if use_temp_gradient
            else temperature_init
        )

        # 计算总能量
        etotal = epot + ekin

        # 仅按 log_interval 打印能量信息，不显示实际温度
        print(f"步 {n:4d} | T目标 = {T_target:6.1f}K | "
              f"势能 = {epot:8.3f}eV | 动能 = {ekin:8.3f}eV | 总能量 = {etotal:8.3f}eV")

    # 按照用户配置的 log_interval 打印能量信息（在下方根据 log_interval 绑定）

    def save_trajectory():
        # 跳过第0步的保存（只从第一步开始记录轨迹）
        if dyn.nsteps == 0:
            return
        # 构造要保存的原子快照并注入元数据
        a_copy = atoms.copy()

        # 计算当前目标温度
        T_target = (
            linear_temperature(dyn.nsteps, steps, temperature_init, temperature_final)
            if use_temp_gradient
            else temperature_init
        )

        # 生成注释字符串：保留 metadata_base 中的键值，添加 Temp
        meta = {} if metadata_base is None else dict(metadata_base)
        meta['Temp'] = float(T_target)

        # 格式化为 key=value（当值包含空格时加引号）
        tokens = []
        for k, v in meta.items():
            sv = str(v)
            if ' ' in sv:
                sv = '"' + sv + '"'
            tokens.append(f"{sv}")
        a_copy.info['Temp'] = ' '.join(tokens)

        trajectory.append(a_copy)

        # 按轨迹间隔输出额外信息
        if dyn.nsteps % traj_interval == 0:
            epot = atoms.get_potential_energy()
            ekin = atoms.get_kinetic_energy()
            print(f"--- 轨迹点 {dyn.nsteps} | 总能量 = {epot + ekin:.3f}eV | 保存结构 ---")

    dyn.attach(save_trajectory, interval=traj_interval)

    print(f"\n{' 分子动力学模拟开始 ':=^80}")
    print(f"系综: {ensemble} | 步长: {timestep_fs}fs | 总步数: {steps}")
    print(f"温度: {temperature_init}K → {temperature_final}K | 压力: {pressure} GPa")
    print(f"轨迹间隔: {traj_interval}步 | 输出间隔: 每 {log_interval} 步（能量按 log_interval 打印）")
    print("-" * 80)
    print("步数   | T目标(K) | 势能(eV)   | 动能(eV)   | 总能量(eV)")
    print("-" * 80)

    # 根据 log_interval 绑定能量打印（若 log_interval <= 0 则不打印）
    if log_interval > 0:
        dyn.attach(print_energy, interval=log_interval)

    # 主模拟循环
    dyn.run(steps)

    # 打印模拟结束信息（不输出最终温度，仅输出最终能量信息）
    epot = atoms.get_potential_energy()
    ekin = atoms.get_kinetic_energy()
    print("-" * 80)
    print(f"模拟完成! 最终势能: {epot:.3f}eV | 最终动能: {ekin:.3f}eV")

    return trajectory


def validate_config(config):
    """增强参数验证"""
    required_keys = [
        'ensemble', 'temperature_init', 'temperature_final', 'pressure',
        'timestep', 'steps', 'traj_interval', 'log_interval'
    ]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"MD 配置缺少以下必需项: {', '.join(missing)}")

    # 温度验证
    if config['temperature_init'] < 0 or config['temperature_final'] < 0:
        raise ValueError("温度不能为负值")

    # 系综验证
    if config['ensemble'] not in ['NVT', 'NPT']:
        raise ValueError(f"不支持的系综类型: {config['ensemble']}")

    # NPT系综压力验证
    if config['ensemble'] == 'NPT' and config['pressure'] < 0:
        raise ValueError("压力不能为负值")


def parse_comment_to_meta(comment):
    """解析 XYZ 注释行中的 key=value 对，返回字典，保留原样字符串值（去掉引号）。"""
    import re
    meta = {}
    if not comment:
        return meta
    # 匹配 key=value 或 key="quoted value"
    pattern = re.compile(r"(\w+)=((?:\"[^\"]*\")|[^\s]+)")
    for m in pattern.finditer(comment):
        key = m.group(1)
        val = m.group(2)
        if val.startswith('"') and val.endswith('"'):
            val = val[1:-1]
        meta[key] = val
    return meta


def main():
    cfg = load_config_from_env()
    if not isinstance(cfg, dict):
        print('错误：MP_CONFIG 格式错误，期望 JSON 对象。')
        sys.exit(2)

    if 'MD' not in cfg:
        print('错误：配置中缺少 "MD" 区块。请在 input.toml 中添加 MD 配置。')
        sys.exit(2)

    md_cfg = cfg['MD']

    # 参数验证
    try:
        validate_config(md_cfg)
    except ValueError as e:
        print(f'配置验证失败: {e}')
        sys.exit(2)

    model_path = cfg.get('model_path') or cfg.get('MODEL_PATH') or 'best-cpu.nequip.pth'
    device = 'cpu'
    calculator = None
    if os.path.exists(model_path) and NequIPCalculator is not None:
        calculator = load_allegro_model(model_path, device=device)
    else:
        if not os.path.exists(model_path):
            print(f'模型文件未找到：{model_path}，将继续但没有能量计算（仅示例）。')

    atoms = create_test_structure()
    # 读取要做 MD 的结构集合（model.xyz）
    model_path = cfg.get('model_xyz', 'model.xyz')
    if not os.path.exists(model_path):
        print(f'错误：未找到结构文件 {model_path}。')
        sys.exit(2)

    try:
        steps = int(md_cfg['steps'])
        timestep = float(md_cfg['timestep'])
        temp_init = float(md_cfg['temperature_init'])
        temp_final = float(md_cfg['temperature_final'])
        traj_interval = int(md_cfg['traj_interval'])
        log_interval = int(md_cfg['log_interval'])
        ensemble = str(md_cfg['ensemble'])
        pressure = float(md_cfg['pressure'])
    except Exception as e:
        print(f'错误：MD 配置类型转换失败或值不合法：{e}')
        sys.exit(2)

    # 读取所有结构帧
    try:
        frames = read(model_path, index=':')
    except Exception as e:
        print(f'错误：读取 {model_path} 失败：{e}')
        sys.exit(2)

    all_trajectory = []
    # 对每个结构执行 MD 并收集轨迹帧
    for idx, frame in enumerate(frames, start=1):
        print(f"\n== 处理第 {idx} 个结构（原子数 {len(frame)}） ==")
        # 提取注释元数据并移除 Lattice, Properties, pbc
        comment = frame.info.get('comment', '')
        meta = parse_comment_to_meta(comment)
        for k in ('Lattice', 'Properties', 'pbc'):
            meta.pop(k, None)

        # 运行 MD（在原子对象上运行前先拷贝，以保留原始 frame 不变）
        a = frame.copy()
        traj = run_md_simulation(
            atoms=a,
            calculator=calculator,
            steps=steps,
            timestep_fs=timestep,
            ensemble=ensemble,
            temperature_init=temp_init,
            temperature_final=temp_final,
            pressure=pressure,
            traj_interval=traj_interval,
            log_interval=log_interval,
            metadata_base=meta
        )

        # 将此结构产生的轨迹追加到总轨迹列表
        if traj:
            all_trajectory.extend(traj)

    # 保存合并轨迹
    if all_trajectory:
        try:
            write('trajectory.xyz', all_trajectory)
            print('所有轨迹已保存到 trajectory.xyz')
        except Exception as e:
            print(f'保存合并轨迹失败：{e}')

    # 保存结果
    try:
        write('final_structure.xyz', atoms)
        print('最终结构已保存到 final_structure.xyz')
    except Exception as e:
        print(f'保存最终结构失败：{e}')

    if all_trajectory:
        try:
            write('trajectory.xyz', all_trajectory)
            print('轨迹已保存到 trajectory.xyz')
        except Exception as e:
            print(f'保存轨迹失败：{e}')


if __name__ == '__main__':
    main()
