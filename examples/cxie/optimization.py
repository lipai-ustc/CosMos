
import os
import json
import sys
import time
import numpy as np
from ase import Atoms, units
from ase.io import write, read
from ase.optimize import BFGS, LBFGS, QuasiNewton
from ase.filters import ExpCellFilter
from ase.calculators.calculator import PropertyNotImplementedError

try:
    from nequip.ase import NequIPCalculator
except Exception:
    NequIPCalculator = None

# 允许的元素列表
ALLOWED_ELEMENTS = ['Al', 'Mg', 'Ti', 'V', 'Cr', 'Fe', 'Co', 'Ni', 'Cu', 'Zr']

def load_config_from_env():
    """从环境变量加载配置"""
    raw = os.environ.get('MP_CONFIG', '{}')
    try:
        cfg = json.loads(raw)
    except Exception:
        cfg = {}
    return cfg

def load_allegro_model(model_path, device='cpu'):
    """加载NequIP模型"""
    if NequIPCalculator is None:
        print('错误：nequip 未安装，无法进行计算。')
        return None
    try:
        calc = NequIPCalculator.from_compiled_model(
            compile_path=model_path,
            device=device,
            species_to_type_name=ALLOWED_ELEMENTS
        )
        print(f'成功加载模型：{model_path}')
        return calc
    except Exception as e:
        print(f'加载模型失败：{e}')
        return None

def parse_comment_to_meta(comment):
    """解析XYZ注释行中的元数据"""
    import re
    meta = {}
    if not comment:
        return meta
    pattern = re.compile(r"(\w+)=((?:\"[^\"]*\")|[^\s]+)")
    for m in pattern.finditer(comment):
        key = m.group(1)
        val = m.group(2)
        if val.startswith('"') and val.endswith('"'):
            val = val[1:-1]
        meta[key] = val
    return meta

def get_cell_angles(cell):
    """计算晶胞角度（度）"""
    a, b, c = np.linalg.norm(cell, axis=1)
    alpha = np.arccos(np.dot(cell[1], cell[2]) / (b * c)) * 180 / np.pi
    beta = np.arccos(np.dot(cell[0], cell[2]) / (a * c)) * 180 / np.pi
    gamma = np.arccos(np.dot(cell[0], cell[1]) / (a * b)) * 180 / np.pi
    return alpha, beta, gamma

def format_cell_info(cell):
    """格式化晶胞信息"""
    lengths = np.linalg.norm(cell, axis=1)
    angles = get_cell_angles(cell)
    return (
        f"a={lengths[0]:.3f}, b={lengths[1]:.3f}, c={lengths[2]:.3f} | "
        f"α={angles[0]:.2f}, β={angles[1]:.2f}, γ={angles[2]:.2f}"
    )

def optimize_with_strainfilter(atoms, calculator, fmax=0.01, max_steps=200, pressure=0.0):
    """先优化原子位置，再用StrainFilter优化晶格和原子位置"""
    atoms.calc = calculator
    # 第一步：只优化原子位置
    pos_optimizer = BFGS(atoms, logfile=None)
    pos_optimizer.run(fmax=fmax, steps=max_steps)
    steps1 = pos_optimizer.get_number_of_steps()
    print(f"位置优化完成，步骤数: {steps1}")


    return atoms,  steps1

def main():
    # 加载配置
    cfg = load_config_from_env()
    if not isinstance(cfg, dict):
        print('错误：MP_CONFIG 格式错误，期望 JSON 对象。')
        sys.exit(2)
    
    # 获取模型路径
    model_path = cfg.get('model_path') or cfg.get('MODEL_PATH') or 'best-cpu.nequip.pth'
    struct_path = cfg.get('model_xyz', 'model.xyz')
    
    # 加载计算器
    device = 'cuda' if cfg.get('use_gpu', False) else 'cpu'
    calculator = load_allegro_model(model_path, device=device)
    if calculator is None:
        print("无法进行计算：计算器未正确加载")
        sys.exit(2)
    
    # 检查结构文件
    if not os.path.exists(struct_path):
        print(f'错误：未找到结构文件 {struct_path}')
        sys.exit(2)
    
    # 优化参数配置（带默认值）
    opt_cfg = cfg.get('Optimization', {})
    fmax = opt_cfg.get('fmax', 0.00001)  # 力收敛标准 (eV/Å)
    steps = opt_cfg.get('max_steps', 2000)  # 最大优化步数
    pressure = opt_cfg.get('pressure', 0.0)  # 优化压力 (GPa)
    optimizer_type = opt_cfg.get('optimizer', 'QuasiNewton')  # 优化器选择
    
    # 读取所有结构
    try:
        frames = read(struct_path, index=':')
        print(f"从 {struct_path} 读取到 {len(frames)} 个结构")
    except Exception as e:
        print(f'读取结构失败：{e}')
        sys.exit(2)
    
    optimized_structures = []
    cell_comparison = []
    optimization_log = []
    
    print(f"\n{' 结构优化开始 ':=^80}")
    print(f"收敛标准: {fmax} | 最大步数: {steps}")
    print("=" * 80)
    
    # 对每个结构进行优化
    for idx, frame in enumerate(frames):
        # 复制结构并记录初始晶胞信息
        atoms = frame.copy()
        initial_cell = atoms.cell.copy()
        initial_volume = atoms.get_volume()
        
        # 解析元数据
        comment = atoms.info.get('comment', '')
        meta = parse_comment_to_meta(comment)
        
        start_time = time.time()
        
        # 运行优化
        try:
            opt_atoms, nsteps = optimize_with_strainfilter(
                atoms=atoms,
                calculator=calculator,
                fmax=fmax,
                max_steps=steps,
                pressure=pressure
            )
        except Exception as e:
            print(f"结构 {idx+1} 优化失败: {e}")
            continue
            
        # 获取优化后晶胞信息
        final_cell = opt_atoms.cell.copy()
        final_volume = opt_atoms.get_volume()
        
        # 计算晶格长度变化
        init_lengths = np.linalg.norm(initial_cell, axis=1)
        final_lengths = np.linalg.norm(final_cell, axis=1)
        length_changes = [(fl - il) / il * 100 for il, fl in zip(init_lengths, final_lengths)]
        
        # 计算晶格角度变化
        init_angles = get_cell_angles(initial_cell)
        final_angles = get_cell_angles(final_cell)
        angle_changes = [fa - ia for ia, fa in zip(init_angles, final_angles)]
        
        # 计算体积变化
        volume_change = (final_volume - initial_volume) / initial_volume * 100
        
        opt_time = time.time() - start_time
        
        # 更新元数据
        meta['initial_cell'] = format_cell_info(initial_cell)
        meta['final_cell'] = format_cell_info(final_cell)
        meta['volume_change'] = f"{volume_change:.2f}%"
        meta['optimizer_steps'] = nsteps
        meta['optimization_time'] = f"{opt_time:.1f}s"
        
        # 添加详细的晶格变化信息
        meta['a_change'] = f"{length_changes[0]:.2f}%"
        meta['b_change'] = f"{length_changes[1]:.2f}%"
        meta['c_change'] = f"{length_changes[2]:.2f}%"
        meta['alpha_change'] = f"{angle_changes[0]:.2f}°"
        meta['beta_change'] = f"{angle_changes[1]:.2f}°"
        meta['gamma_change'] = f"{angle_changes[2]:.2f}°"
        
        # 构造新注释行
        tokens = [f"{k}='{v}'" if ' ' in str(v) else f"{k}={v}" 
                 for k, v in meta.items()]
        opt_atoms.info['comment'] = " ".join(tokens)
        
        # 保存结果
        optimized_structures.append(opt_atoms)
        cell_comparison.append((
            idx, initial_cell, final_cell, 
            length_changes, angle_changes, volume_change
        ))
        
        # 记录日志
        log_entry = (
            f"结构 {idx+1}: {len(opt_atoms)} 原子 | "
            f"优化步数: {nsteps}/{steps} | "
            f"体积变化: {volume_change:.2f}% | "
            f"晶胞变化: a={length_changes[0]:.2f}%, "
            f"b={length_changes[1]:.2f}%, c={length_changes[2]:.2f}% | "
            f"时间: {opt_time:.1f}s"
        )
        optimization_log.append(log_entry)
        print(log_entry)
    
    # 保存优化后的结构
    output_file = cfg.get('output_file', 'optimized_structures.xyz')
    try:
        write('optimized_structures.xyz', optimized_structures, format='extxyz')
        print(f"\n所有优化后的结构已保存到 {output_file}")
    except Exception as e:
        print(f'保存优化结构失败：{e}')
        sys.exit(2)
    
    # 输出体积对比表
    print("\n优化前后晶胞体积对比：")
    print("结构 | 初始体积 | 优化后体积 | 变化 (%)")
    print("-" * 60)
    for item in cell_comparison:
        idx, ic, fc, lc, ac, vc = item
        iv = np.linalg.det(ic)
        fv = np.linalg.det(fc)
        print(f"{idx+1:4d} | {iv:13.3f} | {fv:15.3f} | {vc:8.2f}%")
    
    # 输出晶格参数对比
    print("\n优化前后晶格参数变化：")
    print("结构 | a变化 (%) | b变化 (%) | c变化 (%) | α变化 (°) | β变化 (°) | γ变化 (°)")
    print("-" * 90)
    for item in cell_comparison:
        idx, ic, fc, lc, ac, vc = item
        print(
            f"{idx+1:4d} | {lc[0]:8.2f} | {lc[1]:8.2f} | {lc[2]:8.2f} | "
            f"{ac[0]:8.2f} | {ac[1]:8.2f} | {ac[2]:8.2f}"
        )
    
    # 保存优化日志
    with open('optimization.log', 'w') as f:
        f.write("\n".join(optimization_log))
    print("\n优化日志已保存到 optimization.log")

if __name__ == '__main__':
    main()
