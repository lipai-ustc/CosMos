"""
Al-Cu Binary Nanoparticle Structure Search using EAM Potential
"""
import os
import numpy as np
from ase import Atoms
from ase.calculators.eam import EAM
from cosmos_search import CoSMoSSearch
from ase.data import covalent_radii

# Create Al-Cu binary nanoparticle structure (30 Al, 20 Cu atoms)
# 创建更大的盒子并将原子集中在中心附近
# 增大单元格尺寸至 [30, 30, 30]（原先是 [15, 15, 15]）
atoms = Atoms(['Al']*30 + ['Cu']*20, cell=[30, 30, 30], pbc=True)

# 让原子集中在中心区域（半径为10的球体范围内）
center = np.array([15.0, 15.0, 15.0])  # 盒子中心坐标
radius = 10.0  # 原子分布的球体半径
positions = []

# 获取Al和Cu的共价半径，用于确定最小原子间距
al_radius = covalent_radii[13]  # Al的原子序数是13
cu_radius = covalent_radii[29]  # Cu的原子序数是29
min_distance = (al_radius + cu_radius) * 0.9  # 最小允许距离（共价半径之和的90%）

for i in range(50):
    while True:
        # 生成球坐标系中的随机点
        r = np.random.random() * radius
        theta = np.random.random() * np.pi
        phi = np.random.random() * 2 * np.pi

        # 转换为笛卡尔坐标系
        x = center[0] + r * np.sin(theta) * np.cos(phi)
        y = center[1] + r * np.sin(theta) * np.sin(phi)
        z = center[2] + r * np.cos(theta)
        pos = np.array([x, y, z])

        # 检查与已有原子的距离
        if len(positions) == 0 or all(np.linalg.norm(pos - p) > min_distance for p in positions):
            positions.append(pos)
            break

atoms.positions = np.array(positions)
# Initialize EAM calculator with the local potential file
eam_calc = EAM(potential='AlCu.eam.alloy')
atoms.set_calculator(eam_calc)

# Configure CoSMoS search parameters
search = CoSMoSSearch(
    atoms=atoms,
    H=18,               # Number of hidden states
    w=0.18,             # Weight parameter
    ds=0.28,            # Step size parameter
    max_steps=200       # Maximum search steps
)

# Run the global optimization search
best_structure = search.run()

# Save the optimized structure
best_structure.write('optimized_alcu_nanoparticle.xyz')
print(f"Optimized Al-Cu nanoparticle structure saved to 'optimized_alcu_nanoparticle.xyz'")

# Optional: Visualize the structure
try:
    from ase.visualize import view
    view(best_structure, viewer='x3d')
except ImportError:
    print("ASE visualization not available. Install ASE visualization tools to view the structure.")