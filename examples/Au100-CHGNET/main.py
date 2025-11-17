"""
使用CHGNET预训练力场搜索Au100表面重构结构的示例
"""
import numpy as np
from ase import Atoms
from ase.build import fcc100
from ase.visualize import view
from cosmos_search import CoSMoSSearch

def search_au100_reconstruction_with_chgnet():
    # 创建Au100表面初始结构
    au_surface = fcc100('Au', size=(5, 5, 4), a=4.08, vacuum=10.0)
    
    # 添加随机扰动模拟初始表面状态
    np.random.seed(42)
    positions = au_surface.get_positions()
    positions += np.random.normal(0, 0.2, positions.shape)  # 小幅度随机扰动
    au_surface.set_positions(positions)

    # 初始化CHGNET计算器
    try:
        from chgnet.model import CHGNet
        from chgnet.calculator import CHGNetCalculator
        
        chgnet = CHGNet.load()  # 加载预训练CHGNet模型
        chgnet_calc = CHGNetCalculator(chgnet)
        au_surface.set_calculator(chgnet_calc)
    except ImportError:
        raise ImportError("请安装CHGNET: pip install chgnet")

    # 创建CoSMoS搜索实例
    cosmos = CoSMoSSearch(
        initial_atoms=au_surface,
        calculator=chgnet_calc,
        H=15,               # 高斯势数量
        w=0.15,             # 高斯势高度(eV)
        ds=0.25,            # 步长(Å)
        temperature=400,    # 温度(K)
        mobility_control=True,  # 启用移动性控制
        control_radius=8.0   # 核心区半径(Å)
    )

    # 运行表面重构搜索
    print("开始Au100表面重构搜索...")
    cosmos.run(steps=150)
    print("Au100表面重构搜索完成！")

    # 可视化并保存结果
    final_structure = cosmos.get_best_structure()
    view(final_structure)
    final_structure.write("au100_reconstruction_final.xyz")

    return final_structure

if __name__ == "__main__":
    search_au100_reconstruction_with_chgnet()