import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

class BiasedCalculator(Calculator):
    """
    Biased potential energy calculator for modifying the Potential Energy Surface (PES) during CoSMoS climbing phase
    Adds multiple positive Gaussian bias potentials to the original energy surface to guide structural exploration
    Single Gaussian potential form: V_bias = w * exp(-(d · (R - R1))^2 / (2 * σ^2))
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, base_calculator, mobile_mask, ds, mobile_region, wall_strength, wall_offset):
        super().__init__()
        self.base_calc = base_calculator  # Original potential energy calculator
        self.mobile_mask = mobile_mask  # Boolean mask for mobile atoms
        self.gaussian_params = None  # List of Gaussian parameters, each containing (d, R1, w)
        self.ds = ds  # Step size parameter, used for Gaussian potential width
        self.mobile_region = mobile_region  # Region for mobile atoms
        self.wall_strength = wall_strength  # Strength of wall potential
        self.wall_offset = wall_offset  # Offset of wall potential

    def reset_gaussians(self, gaussian_params):
        """reset gaussian params"""
        self.gaussian_params = gaussian_params
        self.reset()  # reset calculator

    def calculate(self, atoms=None, properties=['energy','forces'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        
        # Get original energy and forces
        self.base_calc.calculate(atoms,properties,system_changes)
        E_base = self.base_calc.results['energy']
        F_base = self.base_calc.results['forces'].flatten()  # (3N,)

        # Get gaussian potential energy and forces
        R = atoms.positions.flatten()  # (3N,)        
        E_gaussian = 0.0
        F_gaussian = np.zeros_like(R)  # (3N,)

        for g_param in self.gaussian_params:
            # g_param should be a tuple or list containing (d, R1, w)
            try:
                d, R1, w = g_param
            except ValueError:
                raise ValueError(f"Invalid Gaussian parameter format: {g_param}")
            dr = R - R1
            # Calculate projection: (R - R1)·Nn
            proj = np.dot(dr, d)
            # Gaussian width uses self.ds (step size); equation (6) in the paper
            E_gaussian += w * np.exp(-(proj**2) / (2 * self.ds**2))
            F_gaussian += w * np.exp(-(proj**2) / (2 * self.ds**2)) * (proj / self.ds**2) * d

        # Total energy and forces are original values plus bias and wall potential contributions
        # Calculate wall potential energy and forces for mobile atoms relative to mobile_region
        E_wall, F_wall = self._calculate_wall_potential(atoms)

        self.results['E_base'] = E_base
        self.results['E_gaussian'] = E_gaussian
        self.results['E_wall'] = E_wall
        self.results['F_base'] = F_base.reshape((-1, 3))
        self.results['F_gaussian'] = F_gaussian.reshape((-1, 3))
        self.results['F_wall'] = F_wall.reshape((-1, 3))

        self.results['energy'] = E_base + E_gaussian + E_wall
        self.results['forces'] = (F_base + F_gaussian + F_wall).reshape((-1, 3))
        self.results['energy_components'] = (E_base, E_gaussian, E_wall)

    def _calculate_wall_potential(self, atoms: Atoms):
        """
        Calculate wall potential energy and forces for mobile atoms relative to mobile_region.
        Returns (wall_energy, wall_forces_flat)
        """
        if self.wall_strength == 0 or self.mobile_region is None:
            return 0.0, np.zeros(3 * len(atoms))
        positions = atoms.get_positions()
        n_atoms = len(atoms)
        wall_energy = 0.0
        wall_forces = np.zeros(3 * n_atoms)

        if self.mobile_region['type'] == 'sphere':
            center = np.array(self.mobile_region.get('center', [0, 0, 0]))
            radius = self.mobile_region.get('radius', 0.0)
            for i in range(n_atoms):
                if not self.mobile_mask[i]:
                    continue
                delta_r = positions[i] - center
                dist = np.linalg.norm(delta_r)
                if dist > radius + self.wall_offset:
                    overshoot = dist - radius - self.wall_offset
                    wall_energy += 0.5 * self.wall_strength * overshoot ** 2
                    direction = delta_r / dist if dist > 0 else np.zeros(3)
                    force = -self.wall_strength * overshoot * direction
                    wall_forces[3*i:3*i+3] = force
        elif self.mobile_region['type'] == 'slab':
            normal = np.array(self.mobile_region.get('normal', [0, 0, 1]))  
            normal = normal / (np.linalg.norm(normal) or 1.0)
            origin = np.array(self.mobile_region.get('origin', [0, 0, 0]))
            min_dist = self.mobile_region.get('min_dist', -5.0)
            max_dist = self.mobile_region.get('max_dist', 5.0)
            for i in range(n_atoms):
                if not self.mobile_mask[i]:
                    continue
                delta_r = positions[i] - origin
                proj_dist = np.dot(delta_r, normal)
                if proj_dist < min_dist - self.wall_offset:
                    overshoot = (min_dist - self.wall_offset) - proj_dist
                    wall_energy += 0.5 * self.wall_strength * overshoot ** 2
                    force = self.wall_strength * overshoot * normal
                    wall_forces[3*i:3*i+3] = force
                elif proj_dist > max_dist + self.wall_offset:
                    overshoot = proj_dist - (max_dist + self.wall_offset)   
                    wall_energy += 0.5 * self.wall_strength * overshoot ** 2
                    force = -self.wall_strength * overshoot * normal
                    wall_forces[3*i:3*i+3] = force
        elif self.mobile_region['type'] in ('lower', 'upper'):
            axis = self.mobile_region.get('axis', 'z').lower()  
            threshold = self.mobile_region.get('threshold', 0.0)
            axis_map = {'x': 0, 'y': 1, 'z': 2}
            axis_index = axis_map.get(axis, 2)
            for i in range(n_atoms):
                if not self.mobile_mask[i]:
                    continue
                coord = positions[i, axis_index]
                if self.mobile_region['type'] == 'lower':
                    # For lower: atoms with coord <= threshold are mobile
                    # Apply wall force if coord > threshold + wall_offset
                    if coord > threshold + self.wall_offset:
                        overshoot = coord - (threshold + self.wall_offset)
                        wall_energy += 0.5 * self.wall_strength * overshoot ** 2
                        force_vec = np.zeros(3)
                        force_vec[axis_index] = -self.wall_strength * overshoot
                        wall_forces[3*i:3*i+3] = force_vec
                else:  # upper
                    # For upper: atoms with coord >= threshold are mobile
                    # Apply wall force if coord < threshold - wall_offset
                    if coord < threshold - self.wall_offset:
                        overshoot = (threshold - self.wall_offset) - coord
                        wall_energy += 0.5 * self.wall_strength * overshoot ** 2
                        force_vec = np.zeros(3)
                        force_vec[axis_index] = self.wall_strength * overshoot
                        wall_forces[3*i:3*i+3] = force_vec
        return wall_energy, wall_forces
