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
