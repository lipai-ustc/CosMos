"""
使用EAM势函数搜索Ni/Fe/Co三元纳米颗粒结构的示例
"""
import numpy as np
from ase import Atoms
from ase.calculators.eam import EAM
from ase.visualize import view
from cosmos_search import CoSMoSSearch

def search_ternary_nanoparticle_with_eam():
    # 创建Ni/Fe/Co三元纳米颗粒的随机初始结构（50个原子）
    np.random.seed(42)
    num_atoms = 50
    symbols = ['Ni', 'Fe', 'Co'] * (num_atoms // 3) + ['Ni'] * (num_atoms % 3)  # 近似等比例混合
    positions = np.random.rand(num_atoms, 3) * 8.0  # 8Å×8Å×8Å盒子内随机分布
    atoms = Atoms(symbols, positions=positions, cell=[12, 12, 12], pbc=True)

    # 初始化EAM计算器
    try:
        # 用户需要准备包含Ni/Fe/Co的EAM势函数文件
        eam_calc = EAM(potential='ternary_ni_fe_co.eam')  # 替换为实际的EAM势函数文件路径
        atoms.set_calculator(eam_calc)
    except FileNotFoundError:
        raise FileNotFoundError("请提供Ni/Fe/Co三元系统的EAM势函数文件")
    except ImportError:
        raise ImportError("请确保ASE已安装EAM模块: pip install ase")

    # 创建CoSMoS搜索实例
    cosmos = CoSMoSSearch(
        initial_atoms=atoms,
        calculator=eam_calc,
        H=18,               # 高斯势数量
        w=0.18,             # 高斯势高度(eV)
        ds=0.28,            # 步长(Å)
        temperature=600,    # 温度(K)，纳米颗粒搜索可适当提高
        mobility_control=True,  # 启用移动性控制
        control_radius=5.5   # 核心区半径(Å)，适合50原子体系
    )

    # 运行纳米颗粒结构搜索
    print("开始Ni/Fe/Co三元纳米颗粒结构搜索...")
    cosmos.run(steps=180)
    print("三元纳米颗粒结构搜索完成！")

    # 可视化并保存结果
    final_structure = cosmos.get_best_structure()
    view(final_structure)
    final_structure.write("ni_fe_co_ternary_nanoparticle.xyz")

    return final_structure

if __name__ == "__main__":
    search_ternary_nanoparticle_with_eam()