import numpy as np
from ase.io import read
from ase.neighborlist import NeighborList
from collections import defaultdict
import os

# 原子半径字典（单位：埃）
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

def analyze_nearest_neighbors(xyz_file, num_neighbors=6, cutoff=10.0):
    """
    分析XYZ文件中所有帧的最近邻原子对，考虑周期性边界条件
    
    参数:
    xyz_file (str): XYZ文件路径
    num_neighbors (int): 每个原子考虑的最近邻数量
    cutoff (float): 邻居列表的截断距离（埃）
    
    返回:
    dict: 原子对类型的统计数据字典
    """
    # 读取所有帧
    try:
        frames = read(xyz_file, index=':')
        print(f"成功读取 {xyz_file}: 共 {len(frames)} 帧")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return {}
    
    # 初始化统计数据收集器
    stats = defaultdict(lambda: {
        'count': 0,
        'distances': [],
        'differences': []
    })
    total_pairs = 0
    
    # 处理每一帧
    for frame_idx, atoms in enumerate(frames):
        # 确保有周期性边界条件
        if not hasattr(atoms, 'pbc'):
            atoms.pbc = True
        if not any(atoms.pbc):
            atoms.pbc = [True, True, True]
        
        # 确定截断半径 - 使用原子半径或统一值
        if all(symbol in ATOMIC_RADII for symbol in set(atoms.symbols)):
            # 如果所有元素都有半径定义，使用原子半径之和
            cutoffs = [0.5 * (ATOMIC_RADII[s] + max(ATOMIC_RADII.values())) for s in atoms.symbols]
        else:
            # 否则使用统一的截断半径
            cutoffs = [cutoff / 2] * len(atoms)
        
        # 创建邻居列表（自动处理周期性边界）
        nl = NeighborList(
            cutoffs, 
            skin=0.0,
            self_interaction=False,
            bothways=True  # 确保每个原子对只计数一次
        )
        nl.update(atoms)
        
        # 收集当前帧的最近邻对
        frame_pairs = []
        
        # 遍历所有原子
        for i in range(len(atoms)):
            # 获取邻居索引和周期性偏移
            indices, offsets = nl.get_neighbors(i)
            
            # 如果没有邻居，跳过
            if len(indices) == 0:
                continue
            
            # 计算所有邻居距离
            distances = [atoms.get_distance(i, int(j), mic=True) for j in indices]
            # 按距离排序并取最近邻
            sorted_indices = np.argsort(distances)
            top_indices = [indices[idx] for idx in sorted_indices[:num_neighbors]]
            top_offsets = [offsets[idx] for idx in sorted_indices[:num_neighbors]]
            top_distances = [distances[idx] for idx in sorted_indices[:num_neighbors]]
            
            for j, offset, dist in zip(top_indices, top_offsets, top_distances):
                # 原子类型
                sym_i = atoms.get_chemical_symbols()[i]
                sym_j = atoms.get_chemical_symbols()[j]
                
                # 原子半径
                r_i = ATOMIC_RADII.get(sym_i, 1.0)
                r_j = ATOMIC_RADII.get(sym_j, 1.0)
                radii_sum = r_i + r_j
                
                # 差异
                diff = dist - radii_sum
                
                # 标准化原子对顺序（按字母顺序）
                pair = tuple(sorted([sym_i, sym_j]))
                pair_key = f"{pair[0]}-{pair[1]}"
                
                # 存储距离和差异
                stats[pair_key]['count'] += 1
                stats[pair_key]['distances'].append(dist)
                stats[pair_key]['differences'].append(diff)
                total_pairs += 1
                
                # 记录原始原子对信息（用于避免重复计数）
                frame_pairs.append((min(i, j), max(i, j)))
        
        # 打印进度
        if (frame_idx + 1) % 10 == 0 or (frame_idx + 1) == len(frames):
            print(f"已处理 {frame_idx + 1}/{len(frames)} 帧, 收集 {total_pairs} 个原子对")
    
    print(f"分析完成: 共处理 {len(frames)} 帧, 收集 {total_pairs} 个原子对")
    return stats

def calculate_final_stats(stats):
    """
    计算最终的统计结果
    
    参数:
    stats (dict): 原子对原始统计数据
    
    返回:
    dict: 包含最终统计数据的字典
    """
    final_stats = {}
    total_pairs = sum(data['count'] for data in stats.values())
    
    for pair, data in stats.items():
        # 计算平均距离和差异
        avg_dist = np.mean(data['distances']) if data['distances'] else 0
        avg_diff = np.mean(data['differences']) if data['differences'] else 0
        min_dist = np.min(data['distances']) if data['distances'] else 0
        max_dist = np.max(data['distances']) if data['distances'] else 0
        std_dist = np.std(data['distances']) if data['distances'] else 0
        
        # 计算百分比
        percentage = (data['count'] / total_pairs) * 100
        
        final_stats[pair] = {
            'count': data['count'],
            'percentage': percentage,
            'avg_distance': avg_dist,
            'min_distance': min_dist,
            'max_distance': max_dist,
            'std_distance': std_dist,
            'avg_difference': avg_diff,
            # 保留原始列表以便后续计算总体统计
            'distances': data['distances'],
            'differences': data['differences']
        }
    
    return final_stats

def save_stats_to_file(stats, filename="neighbor_stats.txt"):
    """
    保存统计结果到文本文件
    
    参数:
    stats (dict): 统计结果字典
    filename (str): 输出文件名
    """
    # 按出现频率排序
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("原子对类型统计结果 (考虑周期性边界条件)\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'原子对':<10} {'数量':>8} {'百分比(%)':>10} {'平均距离(Å)':>12} {'距离范围(Å)':>15} {'标准差':>10} {'平均差异(Å)':>12}\n")
        f.write("-" * 100 + "\n")
        
        for pair, data in sorted_stats:
            dist_range = f"{data['min_distance']:.3f}-{data['max_distance']:.3f}"
            f.write(f"{pair:<10} {data['count']:>8} {data['percentage']:>10.2f} "
                    f"{data['avg_distance']:>12.4f} {dist_range:>15} {data['std_distance']:>10.4f} "
                    f"{data['avg_difference']:>12.4f}\n")
        
        # 添加总体统计
        total_pairs = sum(data['count'] for data in stats.values())
        avg_dist = sum(data['count'] * data['avg_distance'] for data in stats.values()) / total_pairs
        avg_diff = sum(data['count'] * data['avg_difference'] for data in stats.values()) / total_pairs
        
        # 计算总体距离范围和标准差
        all_distances = [d for pair_data in stats.values() for d in pair_data['distances']]
        min_dist_all = min(all_distances) if all_distances else 0
        max_dist_all = max(all_distances) if all_distances else 0
        std_dist_all = np.std(all_distances) if all_distances else 0
        
        f.write("\n总体统计:\n")
        f.write("-" * 100 + "\n")
        f.write(f"总原子对数量: {total_pairs}\n")
        f.write(f"所有原子对平均距离: {avg_dist:.4f} Å\n")
        f.write(f"所有原子对平均差异: {avg_diff:.4f} Å (距离 - 半径和)\n")
        f.write(f"距离范围: {min_dist_all:.4f} - {max_dist_all:.4f} Å\n")
        f.write(f"距离标准差: {std_dist_all:.4f} Å")
    
    print(f"统计结果已保存到文件: {filename}")

def print_summary(stats):
    """
    打印统计摘要到控制台
    
    参数:
    stats (dict): 统计结果字典
    """
    # 按出现频率排序
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True)
    
    print("\n原子对类型统计结果 (考虑周期性边界条件):")
    print("=" * 100)
    print(f"{'原子对':<10} {'数量':>8} {'百分比(%)':>10} {'平均距离(Å)':>12} {'距离范围(Å)':>15} {'标准差':>10} {'平均差异(Å)':>12}")
    print("-" * 100)
    
    for pair, data in sorted_stats:
        dist_range = f"{data['min_distance']:.3f}-{data['max_distance']:.3f}"
        print(f"{pair:<10} {data['count']:>8} {data['percentage']:>10.2f} "
              f"{data['avg_distance']:>12.4f} {dist_range:>15} {data['std_distance']:>10.4f} "
              f"{data['avg_difference']:>12.4f}")
    
    # 添加总体统计
    total_pairs = sum(data['count'] for data in stats.values())
    avg_dist = sum(data['count'] * data['avg_distance'] for data in stats.values()) / total_pairs
    avg_diff = sum(data['count'] * data['avg_difference'] for data in stats.values()) / total_pairs
    
    # 计算总体距离范围和标准差
    all_distances = [d for pair_data in stats.values() for d in pair_data['distances']]
    min_dist_all = min(all_distances) if all_distances else 0
    max_dist_all = max(all_distances) if all_distances else 0
    std_dist_all = np.std(all_distances) if all_distances else 0
    
    print("\n总体统计:")
    print("-" * 100)
    print(f"总原子对数量: {total_pairs}")
    print(f"所有原子对平均距离: {avg_dist:.4f} Å")
    print(f"所有原子对平均差异: {avg_diff:.4f} Å (距离 - 半径和)")
    print(f"距离范围: {min_dist_all:.4f} - {max_dist_all:.4f} Å")
    print(f"距离标准差: {std_dist_all:.4f} Å")

def main():
    # 输入文件
    xyz_file = "model.xyz"  # 替换为您的XYZ文件路径
    
    # 获取原始统计数据
    raw_stats = analyze_nearest_neighbors(xyz_file, num_neighbors=6)
    
    if not raw_stats:
        print("未找到有效数据，程序退出")
        return
    
    # 计算最终统计
    final_stats = calculate_final_stats(raw_stats)
    
    # 打印摘要
    print_summary(final_stats)
    
    # 保存详细结果到文件
    save_stats_to_file(final_stats, "model_neighbor_stats.txt")
    
    print("\n分析完成! 详细统计已保存到 model_neighbor_stats.txt")

if __name__ == "__main__":
    main()
