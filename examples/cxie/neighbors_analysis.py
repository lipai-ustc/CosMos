import numpy as np
from ase.io import read, write
from ase import Atoms
from collections import defaultdict
import os
import itertools
from scipy.spatial.distance import cdist

# 原子半径字典（扩展版）
ATOMIC_RADII = {
    'H': 0.53, 'He': 0.31,
    'Li': 1.67, 'Be': 1.12, 'B': 0.87, 'C': 0.67, 'N': 0.56, 'O': 0.48, 'F': 0.42, 'Ne': 0.38,
    'Na': 1.90, 'Mg': 1.45, 'Al': 1.18, 'Si': 1.11, 'P': 0.98, 'S': 0.88, 'Cl': 0.79, 'Ar': 0.71,
    'K': 2.43, 'Ca': 1.94, 'Sc': 1.48, 'Ti': 1.32, 'V': 1.22, 'Cr': 1.18, 'Mn': 1.17, 'Fe': 1.17,
    'Co': 1.16, 'Ni': 1.15, 'Cu': 1.17, 'Zn': 1.25, 'Ga': 1.26, 'Ge': 1.22, 'As': 1.20, 'Se': 1.16,
    'Br': 1.14, 'Kr': 1.10,
    'Rb': 2.65, 'Sr': 2.19, 'Y': 1.82, 'Zr': 1.60, 'Nb': 1.47, 'Mo': 1.39, 'Tc': 1.37, 'Ru': 1.35,
    'Rh': 1.34, 'Pd': 1.38, 'Ag': 1.44, 'Cd': 1.49, 'In': 1.55, 'Sn': 1.58, 'Sb': 1.60, 'Te': 1.60,
    'I': 1.40, 'Xe': 1.30,
    'Cs': 2.98, 'Ba': 2.53, 'La': 1.95, 'Ce': 1.85, 'Pr': 2.47, 'Nd': 2.06, 'Pm': 2.05, 'Sm': 2.38,
    'Eu': 2.31, 'Gd': 2.33, 'Tb': 2.25, 'Dy': 2.28, 'Ho': 2.26, 'Er': 2.26, 'Tm': 2.22, 'Yb': 2.22,
    'Lu': 2.17, 'Hf': 1.58, 'Ta': 1.46, 'W': 1.39, 'Re': 1.37, 'Os': 1.35, 'Ir': 1.36, 'Pt': 1.38,
    'Au': 1.44, 'Hg': 1.52, 'Tl': 1.71, 'Pb': 1.75, 'Bi': 1.82, 'Po': 1.77, 'At': 1.74, 'Rn': 1.72,
    'U': 1.50, 'Th': 1.79, 'Pa': 1.63, 'Np': 1.56, 'Pu': 1.58, 'Am': 1.70, 'Cm': 1.71, 'Bk': 1.70,
    'Cf': 1.80, 'Es': 1.90, 'Fm': 2.00, 'Md': 2.10, 'No': 2.20, 'Lr': 2.30,
    'Fr': 3.00, 'Ra': 2.70,
    'Ac': 2.00, 'Db': 1.40, 'Sg': 1.30, 'Bh': 1.30, 'Hs': 1.30, 'Mt': 1.30, 'Ds': 1.30, 'Rg': 1.30,
    'Cn': 1.30
}

def create_periodic_supercell(atoms):
    """
    创建3×3×3超晶胞以正确考虑周期性边界条件
    
    参数:
    atoms (Atoms): ASE原子对象
    
    返回:
    Atoms: 扩展后的超晶胞原子对象
    """
    # 创建超晶胞
    scaled_positions = atoms.get_scaled_positions()
    symbols = atoms.get_chemical_symbols()
    
    # 生成3×3×3超晶胞偏移
    offsets = list(itertools.product([-1, 0, 1], repeat=3))
    
    # 创建超晶胞的原子位置和符号
    supercell_positions = []
    supercell_symbols = []
    
    # 原始晶胞原子
    for i, pos in enumerate(scaled_positions):
        supercell_positions.append(pos)
        supercell_symbols.append(symbols[i])
    
    # 镜像原子
    for offset in offsets:
        if offset == (0, 0, 0):  # 跳过中心晶胞（已经添加）
            continue
            
        for i, pos in enumerate(scaled_positions):
            # 计算镜像位置
            mirrored_pos = pos + np.array(offset)
            supercell_positions.append(mirrored_pos)
            supercell_symbols.append(symbols[i])
    
    # 创建新的原子对象
    supercell = Atoms(
        symbols=supercell_symbols,
        cell=atoms.cell,
        pbc=True,
        scaled_positions=supercell_positions
    )
    
    return supercell

def get_nearest_neighbors(xyz_file, frame_indices, n_neighbors, periodic=False):
    """
    获取多个指定帧结构中每帧的前n个最近邻原子对的信息
    
    参数:
    xyz_file (str): XYZ文件路径
    frame_indices (list): 要分析的帧索引列表(0-based)
    n_neighbors (int): 每帧要获取的最近邻原子对数量
    periodic (bool): 是否考虑周期性边界条件
    
    返回:
    tuple: 包含两个元素的元组
           1. dict: 以帧索引为键，值为最近邻原子对信息列表的字典
           2. list: 所有指定帧的ASE原子对象列表
    """
    # 读取XYZ文件
    try:
        atoms_list = read(xyz_file, index=':')
    except Exception as e:
        raise IOError(f"读取文件 {xyz_file} 失败: {str(e)}")
    
    total_frames = len(atoms_list)
    selected_frames = []  # 保存所有指定帧的原子对象
    
    # 检查帧索引是否有效
    invalid_indices = [idx for idx in frame_indices if idx >= total_frames or idx < 0]
    if invalid_indices:
        raise ValueError(f"无效帧索引: {invalid_indices}。文件包含 {total_frames} 帧，索引范围为 0 到 {total_frames-1}。")
    
    results = {}
    
    for frame_index in frame_indices:
        print(f"处理帧 {frame_index}...")
        original_atoms = atoms_list[frame_index]
        selected_frames.append(original_atoms.copy())  # 保存原始帧
        
        # 如果需要考虑周期性边界条件
        if periodic:
            # 创建3×3×3超晶胞
            atoms = create_periodic_supercell(original_atoms)
            
            # 仅保留中心晶胞原子（原始原子）
            center_atoms = atoms.copy()
            center_atoms = center_atoms[:len(original_atoms)]
        else:
            atoms = original_atoms.copy()
            center_atoms = atoms.copy()
        
        # 获取原子位置和元素符号
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        
        # 获取中心晶胞原子位置
        center_positions = center_atoms.get_positions()
        
        # 计算所有原子对的距离并存储
        distances = []
        
        # 使用高效的cdist计算距离矩阵
        dist_matrix = cdist(center_positions, positions)
        
        # 找到每个中心原子的最近邻
        for i, center_pos in enumerate(center_positions):
            # 排除自身距离（对角线）
            valid_indices = np.where(np.arange(len(positions)) != i)[0]
            min_indices = np.argsort(dist_matrix[i, valid_indices])[:n_neighbors]
            
            for idx in min_indices:
                j = valid_indices[idx]
                distance = dist_matrix[i, j]
                
                # 获取原子类型和半径
                symbol_i = symbols[i]  # 中心原子（原始晶胞）
                symbol_j = symbols[j]   # 近邻原子（可能来自镜像晶胞）
                radius_i = ATOMIC_RADII.get(symbol_i, 1.0)
                radius_j = ATOMIC_RADII.get(symbol_j, 1.0)
                radius_sum = radius_i + radius_j
                
                # 存储距离和相关信息
                distances.append((distance, symbol_i, symbol_j, radius_i, radius_j, radius_sum))
        
        # 按距离排序并获取前n_neighbors个最近邻
        distances.sort(key=lambda x: x[0])
        results[frame_index] = distances[:n_neighbors]
    
    return results, selected_frames

def analyze_neighbors(neighbors_dict, selected_frames, output_file=None):
    """
    分析并输出多个帧的最近邻原子对信息，并保存选定帧到XYZ文件
    
    参数:
    neighbors_dict (dict): 最近邻原子对信息字典，键为帧索引，值为最近邻原子对列表
    selected_frames (list): 所有选定帧的ASE原子对象列表
    output_file (str): 输出文件路径(可选)
    """
    # 创建输出目录（如果需要）
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # 准备输出内容
    output_lines = []
    
    # 总体统计
    total_pair_counter = defaultdict(int)
    frame_count = len(neighbors_dict)
    total_neighbors = 0
    
    # 保存选定帧到单独的XYZ文件
    frames_filename = "selected_frames.xyz"
    try:
        write(frames_filename, selected_frames)
        output_lines.append(f"所有选定帧已保存到文件: {frames_filename}\n")
        print(f"\n所有选定帧已保存到文件: {frames_filename}")
    except Exception as e:
        error_msg = f"保存选定帧失败: {str(e)}"
        output_lines.append(error_msg + "\n")
        print(error_msg)
    
    # 遍历每一帧的结果
    for i, frame_index in enumerate(frame_indices):
        neighbors = neighbors_dict[frame_index]
        # 获取当前帧的原子对象
        current_frame = selected_frames[i]
        
        # 每帧的统计
        frame_pair_counter = defaultdict(int)
        
        # 添加帧标识
        frame_header = f"\n帧 {frame_index} 的最近邻原子对分析结果:"
        print(frame_header)
        output_lines.append(frame_header)
        separator = "=" * 80
        print(separator)
        output_lines.append(separator)
        
        # 表头
        header = f"{'原子对':<10} {'距离(Å)':<10} {'半径和(Å)':<12} {'类型1':<8} {'类型2':<8} {'半径1':<8} {'半径2':<8} {'差异(%)':<8}"
        print(header)
        output_lines.append(header)
        
        sub_separator = "-" * 80
        print(sub_separator)
        output_lines.append(sub_separator)
        
        # 处理每个原子对
        for i, (dist, sym1, sym2, r1, r2, r_sum) in enumerate(neighbors):
            # 标准化原子对顺序(按字母顺序)
            pair_key = tuple(sorted([sym1, sym2]))
            pair_str = f"{sym1}-{sym2}"
            
            # 计算差异百分比
            diff_percent = (dist - r_sum) / r_sum * 100
            
            # 更新计数器
            frame_pair_counter[pair_key] += 1
            total_pair_counter[pair_key] += 1
            total_neighbors += 1
            
            # 格式化输出行
            result_line = f"{pair_str:<10} {dist:<10.4f} {r_sum:<12.4f} {sym1:<8} {sym2:<8} {r1:<8.4f} {r2:<8.4f} {diff_percent:<8.2f}"
            print(result_line)
            output_lines.append(result_line)
        
        # 输出该帧的原子对统计
        frame_stats = f"\n帧 {frame_index} 原子对类型统计:"
        print(frame_stats)
        output_lines.append(frame_stats)
        
        stats_separator = "-" * 30
        print(stats_separator)
        output_lines.append(stats_separator)
        
        for pair_key, count in sorted(frame_pair_counter.items(), key=lambda x: x[1], reverse=True):
            pair_str = f"{pair_key[0]}-{pair_key[1]}"
            stat_line = f"{pair_str}: {count} 对"
            print(stat_line)
            output_lines.append(stat_line)
    
    # 总体统计
    overall_header = f"\n总体统计 (共分析 {frame_count} 帧, {total_neighbors} 个原子对):"
    print(overall_header)
    output_lines.append(overall_header)
    
    overall_separator = "-" * 40
    print(overall_separator)
    output_lines.append(overall_separator)
    
    # 按出现频率排序
    for pair_key, count in sorted(total_pair_counter.items(), key=lambda x: x[1], reverse=True):
        pair_str = f"{pair_key[0]}-{pair_key[1]}"
        percentage = (count / total_neighbors) * 100
        overall_line = f"{pair_str}: {count} 对 ({percentage:.1f}%)"
        print(overall_line)
        output_lines.append(overall_line)
    
    # 平均距离统计
    avg_distances = defaultdict(list)
    for neighbors in neighbors_dict.values():
        for dist, sym1, sym2, r1, r2, r_sum in neighbors:
            pair_key = tuple(sorted([sym1, sym2]))
            avg_distances[pair_key].append(dist)
    
    print("\n平均距离统计:")
    output_lines.append("\n平均距离统计:")
    
    for pair_key, dist_list in sorted(avg_distances.items(), key=lambda x: np.mean(x[1])):
        pair_str = f"{pair_key[0]}-{pair_key[1]}"
        avg_dist = np.mean(dist_list)
        std_dist = np.std(dist_list)
        min_dist = np.min(dist_list)
        max_dist = np.max(dist_list)
        avg_line = f"{pair_str}: {avg_dist:.4f} ± {std_dist:.4f} Å (范围: {min_dist:.4f}-{max_dist:.4f} Å)"
        print(avg_line)
        output_lines.append(avg_line)
    
    # 输出到文件(如果指定了输出文件)
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines))
        print(f"\n完整分析结果已保存到文件: {output_file}")


if __name__ == '__main__':
    # 示例用法
    xyz_file = "model.xyz"  # 替换为您的XYZ文件路径
    frame_indices = [140, 229, 374, 499, 575, 691, 744, 815, 819, 837, 847, 876, 980, 1090, 1174]  # 要分析的多帧索引
    n_neighbors = 5  # 每帧要获取的最近邻原子对数量
    output_filename = "neighbors_analysis.txt"
    
    # 获取最近邻原子对和选定帧
    try:
        neighbors_dict, selected_frames = get_nearest_neighbors(
            xyz_file, 
            frame_indices, 
            n_neighbors, 
            periodic=True
        )
    except Exception as e:
        print(f"错误: {str(e)}")
        exit(1)
    
    # 分析并输出结果
    analyze_neighbors(neighbors_dict, selected_frames, output_file=output_filename)
    
    # 可选：也可以单独保存所有选定帧
    print("\n分析完成!")
