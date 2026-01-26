import os
import json
import random
import numpy as np
import math
from collections import Counter
from itertools import combinations, permutations
from ase import Atoms
from ase.io import write

def load_mp_config():
    cfg_json = os.environ.get('MP_CONFIG')
    if not cfg_json:
        return {}
    try:
        return json.loads(cfg_json)
    except Exception:
        return {}

def validate_and_generate(block):
    """结构生成逻辑：遍历所有元素组合和所有原子分配排列方案"""
    # 参数解析与校验
    try:
        element_classes = str(block.get('element_classes', '')).split()
        total_atoms = int(block.get('total_atoms', 0))
        num_element_types = int(block.get('num_element_types', 0))
    except Exception as e:
        return False, f"参数解析错误: {e}"
    
    # 校验逻辑
    if not element_classes:
        return False, 'element_classes 不能为空'
    if total_atoms <= 0:
        return False, 'total_atoms 必须大于 0'
    if num_element_types <= 0:
        return False, 'num_element_types 必须大于 0'
    if num_element_types > len(element_classes):
        return False, 'num_element_types 必须小于等于 element_classes 的数量'
    if num_element_types > total_atoms:
        return False, 'num_element_types 必须小于 total_atoms'

    # 准备输出文件
    output = block.get('output', 'all_structures.exyz')
    
    # 原子半径定义（用于确定最小距离）
    element_radii = {
        'H': 0.53, 'He': 0.31,
        'Li': 1.67, 'Be': 1.12, 'B': 0.87, 'C': 0.67, 'N': 0.56, 'O': 0.48, 'F': 0.42, 'Ne': 0.38,
        'Na': 1.90, 'Mg': 1.45, 'Al': 1.18, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
        'K': 2.43, 'Ca': 1.94, 'Sc': 1.84, 'Ti': 1.76, 'V': 1.71, 'Cr': 1.66, 'Mn': 1.61, 'Fe': 1.56,
        'Co': 1.52, 'Ni': 1.49, 'Cu': 1.45, 'Zn': 1.42, 'Ga': 1.36, 'Ge': 1.25, 'As': 1.14, 'Se': 1.03,
        'Br': 0.94, 'Kr': 0.88
    }
    
    # 生成参数
    max_cell_attempts = 50  # 最大晶胞尝试次数
    max_position_attempts = 500  # 最大位置尝试次数
    
    # 计算原子密度相关参数 - 增大晶胞尺寸防止过于紧凑
    n = total_atoms
    density_factor = n**(1/3) * 1.8  # 从1.3增加到1.8
    length_min = max(3, 2 * density_factor)  # 从2增加到3
    length_max = 7.5 * density_factor  # 从7增加到10
    
    # 体积安全系数 - 显著增大以防止结构过于紧凑
    if n <= 4:
        k_v = 35.0  # 从15.0增加到35.0
    elif n <= 10:
        k_v = 30.0  # 从12.0增加到30.0
    elif n <= 20:
        k_v = 25.0
    else:
        k_v = 20.0  # 从8.0增加到20.0
    v_min = n * k_v
    
    # 统计信息
    success_count = 0
    failed_combinations = set()
    total_configurations = 0
    
    # 初始化输出文件（清空）
    open(output, 'w').close()
    
    # 周期性距离计算函数 - 正确实现周期性边界条件
    def periodic_distance(frac1, frac2, cell):
        """计算两个分数坐标之间的最小周期距离"""
        delta_frac = frac1 - frac2
        # 将差值折叠到[-0.5, 0.5]区间以找到最近镜像
        delta_frac -= np.round(delta_frac)
        delta_cart = delta_frac @ cell
        return np.linalg.norm(delta_cart)
    
    # 生成所有可能的原子分配方案
    def generate_atom_allocations(n, k):
        """生成所有可能的原子分配方案（包括所有排列）"""
        partitions = []
        def find_partitions(remaining, parts, min_val=1):
            if len(parts) == k - 1:
                last_part = remaining
                if last_part >= min_val:
                    partitions.append(parts + [last_part])
                return
            for i in range(min_val, remaining - min_val * (k - len(parts) - 1) + 1):
                find_partitions(remaining - i, parts + [i], min_val)
        
        find_partitions(n, [], 1)
        
        all_allocations = set()
        for p in partitions:
            for perm in set(permutations(p)):
                all_allocations.add(perm)
        
        return sorted(all_allocations)
    
    # 遍历所有元素组合
    for combination_idx, chosen_elements in enumerate(combinations(element_classes, num_element_types)):
        print(f"正在处理组合 {combination_idx+1}/{len(list(combinations(element_classes, num_element_types)))}: {'-'.join(chosen_elements)}")
        
        # 1. 计算该组合的最小距离 - 基于元素半径
        min_distance = 1.5 * sum(element_radii.get(e, 1.0) for e in chosen_elements) / len(chosen_elements)
        min_distance = max(1.8, min_distance)  # 确保最小距离不小于1.8Å
        
        # 2. 生成该元素组合下所有可能的原子分配方案
        allocations = generate_atom_allocations(total_atoms, num_element_types)
        total_configurations += len(allocations)
        print(f"  该组合有 {len(allocations)} 种原子分配方案, 最小距离设置为: {min_distance:.2f} Å")
        
        # 遍历所有原子分配方案
        for alloc_idx, base_counts in enumerate(allocations):
            if alloc_idx % 10 == 0 or alloc_idx == len(allocations) - 1:
                print(f"    处理分配方案 {alloc_idx+1}/{len(allocations)}: {'/'.join(f'{e}:{c}' for e, c in zip(chosen_elements, base_counts))}")
            
            valid_structure = False
            positions = []
            
            for cell_attempt in range(max_cell_attempts):
                # 阶段1：晶胞参数生成
                while True:
                    lengths = [
                        random.uniform(length_min, length_max),
                        random.uniform(length_min, length_max),
                        random.uniform(length_min, length_max)
                    ]
                    max_len, min_len = max(lengths), min(lengths)
                    if max_len/min_len >= 5:  # 从10减少到5，避免极端比例
                        continue
                    
                    angles = [
                        random.uniform(60.0, 120.0),
                        random.uniform(60.0, 120.0),
                        random.uniform(60.0, 120.0)
                    ]
                    
                    cos_vals = [math.cos(math.radians(a)) for a in angles]
                    vol_factor = 1 + 2*cos_vals[0]*cos_vals[1]*cos_vals[2] 
                    vol_factor -= cos_vals[0]**2 + cos_vals[1]**2 + cos_vals[2]**2
                    if vol_factor <= 1e-6:  # 避免数值精度问题
                        continue
                    
                    volume = lengths[0] * lengths[1] * lengths[2] * math.sqrt(vol_factor)
                    if volume >= v_min:
                        break
                
                # 构建晶胞矩阵
                a, b, c = lengths
                alpha_r, beta_r, gamma_r = [math.radians(a) for a in angles]
                
                vx = np.array([a, 0.0, 0.0])
                vy = np.array([b * math.cos(gamma_r), b * math.sin(gamma_r), 0.0])
                vz = np.array([
                    c * math.cos(beta_r),
                    c * (math.cos(alpha_r) - math.cos(beta_r)*math.cos(gamma_r)) / math.sin(gamma_r),
                    c * math.sqrt(max(0, 1 - math.cos(beta_r)**2 - 
                                 ((math.cos(alpha_r)-math.cos(beta_r)*math.cos(gamma_r))/math.sin(gamma_r))**2))
                ])
                cell = np.stack([vx, vy, vz], axis=0)
                
                # 阶段2：原子位置生成（使用分数坐标确保原子在晶胞内）
                existing_fracs = []  # 存储分数坐标
                positions = []       # 存储笛卡尔坐标
                position_found = True
                
                for atom_idx in range(total_atoms):
                    atom_valid = False
                    
                    for pos_attempt in range(max_position_attempts):
                        # 在晶胞内生成随机分数坐标（确保原子在晶格内）
                        new_frac = np.random.rand(3)
                        
                        # 第一个原子直接接受
                        if not existing_fracs:
                            existing_fracs.append(new_frac)
                            positions.append(new_frac @ cell)
                            atom_valid = True
                            break
                        
                        # 计算与所有已有原子的最小周期距离
                        min_dist = min(periodic_distance(new_frac, ef, cell) for ef in existing_fracs)
                        
                        if min_dist > min_distance:
                            existing_fracs.append(new_frac)
                            positions.append(new_frac @ cell)
                            atom_valid = True
                            break
                    
                    if not atom_valid:
                        position_found = False
                        break
                
                if position_found:
                    valid_structure = True
                    break
            
            if not valid_structure:
                config_id = f"{'-'.join(chosen_elements)}__{'_'.join(map(str, base_counts))}"
                print(f"    无法生成有效结构: {config_id}")
                failed_combinations.add(config_id)
                continue
            
            # 3. 生成元素符号序列
            symbols = []
            for element, count in zip(chosen_elements, base_counts):
                symbols.extend([element] * count)
            
            # 随机打乱元素顺序
            combined = list(zip(symbols, positions))
            random.shuffle(combined)
            symbols, positions = zip(*combined)
            positions = np.array(positions)
            
            # 4. 使用ASE写入EXYZ格式
            try:
                element_str = '-'.join(chosen_elements)
                counts_str = '/'.join(f"{e}:{c}" for e, c in zip(chosen_elements, base_counts))
                
                atoms = Atoms(
                    symbols=symbols,
                    positions=positions,
                    cell=cell,
                    pbc=True
                )
                
                atoms.info = {
                    'Combination': element_str,
                    'Element_Counts': counts_str,
                    'Total_Atoms': total_atoms,
                    'Num_Element_Types': num_element_types,
                    'Allocation_ID': f"{'_'.join(map(str, base_counts))}",
                    'Min_Distance': f"{min_distance:.3f}"
                }
                
                write(output, atoms, format='extxyz', append=True)
                success_count += 1
            except Exception as e:
                config_id = f"{'-'.join(chosen_elements)}__{'_'.join(map(str, base_counts))}"
                failed_combinations.add(config_id)
                print(f"    写出结构失败: {str(e)}")
    
    # 返回结果
    result_msg = f"完成 {success_count}/{total_configurations} 个配置的结构生成 -> {output}"
    if failed_combinations:
        result_msg += f"\n失败配置: {', '.join(list(failed_combinations)[:5])}{'...' if len(failed_combinations) > 5 else ''}"
    
    return True, result_msg

def main():
    config = load_mp_config()
    structures_block = config.get('Structures', {})
    
    print('结构生成器启动 - 已改进算法解决结构紧凑和周期性边界问题')
    print('收到 Structures 配置:')
    if structures_block:
        for k, v in structures_block.items():
            print(f'  {k}: {v}')
    else:
        print('  (未在 MP_CONFIG 中找到 Structures 块)')
    
    print('\n正在使用改进算法生成结构:')
    ok, msg = validate_and_generate(structures_block)
    if not ok:
        print(f'参数校验失败：{msg}')
    else:
        print(f'生成成功：{msg}')

if __name__ == '__main__':
    main()
