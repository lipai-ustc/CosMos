import numpy as np


def rotate_vector(v, axis, angle):
    """
    Rotate vector using Rodrigues' rotation formula
    Reference: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    Parameters:
    v: Vector to be rotated
    axis: Rotation axis vector
    angle: Rotation angle (radians)

    Returns:
    Rotated vector
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
    """
    Automatically load different types of potential calculators based on configuration
    """
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
        # Parse LAMMPS potential configuration
        lammps_commands = potential_config['commands']
        return LAMMPSlib(lmpcmds=lammps_commands)
    else:
        raise ValueError(f"Unsupported potential type: {pot_type}")
