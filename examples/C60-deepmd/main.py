"""
使用DeepMD势函数搜索C60结构的示例
"""
import numpy as np
from ase import Atoms
from ase.visualize import view
from cosmos_search import CoSMoSSearch

def search_c60_with_deepmd():
    # 随机生成60个碳原子的初始结构
    np.random.seed(42)
    positions = np.random.rand(60, 3) * 10.0  # 10Å×10Å×10Å盒子内随机分布
    atoms = Atoms('C60', positions=positions, cell=[15, 15, 15], pbc=True)

    # 初始化DeepMD计算器
    try:
        from deepmd.calculator import DP
        dp_calc = DP(model="path/to/your/deepmd/model")  # 用户需替换为实际模型路径
        atoms.set_calculator(dp_calc)
    except ImportError:
        raise ImportError("请安装DeepMD-kit: pip install deepmd-kit")

    # 创建CoSMoS搜索实例
    cosmos = CoSMoSSearch(
        initial_atoms=atoms,
        calculator=dp_calc,
        H=20,               # 高斯势数量
        w=0.2,              # 高斯势高度(eV)
        ds=0.3,             # 步长(Å)
        temperature=500,    # 温度(K)
        mobility_control=True,  # 启用移动性控制
        control_radius=6.0   # 核心区半径(Å)
    )

    # 运行搜索
    print("开始C60结构搜索...")
    cosmos.run(steps=200)
    print("C60结构搜索完成！")

    # 可视化并保存结果
    final_structure = cosmos.get_best_structure()
    view(final_structure)
    final_structure.write("c60_final_structure.xyz")

    return final_structure

if __name__ == "__main__":
    search_c60_with_deepmd()