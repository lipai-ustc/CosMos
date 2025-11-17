import numpy as np


def rotate_vector(v, axis, angle):
    """使用罗德里格斯公式旋转向量
    参考: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    
    参数:
        v: 待旋转的向量
        axis: 旋转轴向量
        angle: 旋转角度(弧度)
    
    返回:
        旋转后的向量
    """
    v = np.asarray(v)
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    cross = np.cross(axis, v)
    dot = np.dot(axis, v)
    return v * cos_theta + cross * sin_theta + axis * dot * (1 - cos_theta)


import importlib
from ase.calculators import eam

def load_potential(potential_config):
    """根据配置自动加载不同类型的势函数计算器"""
    pot_type = potential_config['type'].lower()
    
    if pot_type == 'eam':
        from ase.calculators.eam import EAM
        return EAM(potential=potential_config['file'])
    elif pot_type == 'chgnet':
        from chgnet.model import CHGNet
        model = CHGNet.load()
        return model.calculator()
    elif pot_type == 'deepmd':
        from deepmd.calculator import DP
        return DP(model=potential_config['model'])
    elif pot_type == 'lammps':
        from ase.calculators.lammpslib import LAMMPSlib
        # 解析LAMMPS势函数配置
        lammps_commands = potential_config['commands']
        return LAMMPSlib(lmpcmds=lammps_commands)
    else:
        raise ValueError(f"Unsupported potential type: {pot_type}")
