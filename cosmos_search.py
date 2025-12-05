# cosmos_search.py (v2: faithful to Shang & Liu 2013)
# Reference 1: Shang, R., & Liu, J. (2013). Stochastic surface walking method for global optimization of atomic clusters and biomolecules. The Journal of Chemical Physics, 139(24), 244104.
# Reference 2: J. Chem. Theory Comput. 2012, 8, 2215
import os
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from cosmos_utils import is_duplicate_by_desc_and_energy, write_structure_with_energy

class BiasedCalculator(Calculator):
    """
    Biased potential energy calculator for modifying the Potential Energy Surface (PES) during CoSMoS climbing phase
    Adds multiple positive Gaussian bias potentials to the original energy surface to guide structural exploration
    Single Gaussian potential form: V_bias = w * exp(-(d · (R - R1))^2 / (2 * σ^2))
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, base_calculator, gaussian_params, ds=0.2, wall_energy=0.0, wall_forces=None):
        super().__init__()
        self.base_calc = base_calculator  # Original potential energy calculator
        self.gaussian_params = gaussian_params  # List of Gaussian parameters, each containing (d, R1, w)
        self.ds = ds  # Step size parameter, used for Gaussian potential width
        self.wall_energy = wall_energy  # Pre-calculated wall potential energy
        self.wall_forces = wall_forces if wall_forces is not None else np.array([])  # Pre-calculated wall forces

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        
        # Get original energy and forces
        atoms_base = atoms.copy()
        atoms_base.calc = self.base_calc
        E0 = atoms_base.get_potential_energy()
        F0 = atoms_base.get_forces().flatten()  # (3N,)

        # Current position
        R = atoms.positions.flatten()  # (3N,)

        # Initialize total bias energy and forces to 0
        V_bias_total = 0.0
        F_bias_total = np.zeros_like(R)

        # Sum all Gaussian potentials - according to equations (5) and (6) in the paper
        for g_param in self.gaussian_params:
            # g_param should be a tuple or list containing (d, R1, w)
            if len(g_param) != 3:
                continue
                
            d, R1, w = g_param
            R1_flat = R1.flatten()
            dr = R - R1_flat
            # Calculate projection: (R - R1)·Nn
            proj = np.dot(dr, d)
            
            # Parameter a in equation (6) of the paper
            a = 2.0  # Controls Gaussian potential width and strength
            
            # Calculate energy contribution of single Gaussian potential - according to equation (5) in the paper
            V_bias = w * np.exp(-(proj**2) / (2 * self.ds**2))
            V_bias_total += V_bias
            
            # Calculate force contribution of single Gaussian potential - derived from equation (5) in the paper
            # F = -dV/dR = w * exp(...) * (proj / ds²) * d
            F_bias = w * np.exp(-(proj**2) / (2 * self.ds**2)) * (proj / self.ds**2) * d
            F_bias_total += F_bias

        # Total energy and forces are original values plus bias and wall potential contributions
        # Ensure wall_forces has correct shape
        wall_forces = self.wall_forces if len(self.wall_forces) == len(R) else np.zeros_like(R)
        self.results['energy'] = E0 + V_bias_total + self.wall_energy
        self.results['forces'] = (F0 + F_bias_total + wall_forces).reshape((-1, 3))


class CoSMoSSearch:
    def __init__(
        self,
        initial_atoms: Atoms,
        calculator,
        soap_species=None,
        ds=0.2,              # Step size (Å) also Gaussian potential width
        duplicate_tol=0.01,
        fmax=0.05,
        max_steps=500,
        output_dir="cosmos_output",
        H=20,                # Number of Gaussian potentials
        w=0.2,               # Gaussian potential height (eV)
        temperature=500,     # Temperature (K) for Metropolis criterion
        # New parameters to align with cosmos_run.py
        radius=5.0,          # Core region radius
        decay=0.99,          # Decay coefficient
        wall_strength=0.0,   # Wall potential strength (eV/Å²)
        wall_offset=2.0,     # Wall potential distance offset (Å)
        # Mobility control parameters - simplified design
        mobility_control=None,            # Mobility control configuration dict or None
        **kwargs                      # Additional parameters
    ):
        # Initialize minimum pool
        self.pool = []  # Stores found minimum energy structures
        self.atoms = initial_atoms.copy()
        self.base_calc = calculator  # Calculator for real potential energy
        self.soap_species = soap_species or list(set(initial_atoms.get_chemical_symbols()))
        self.ds = ds  # Step size parameter controlling structure movement distance
        self.duplicate_tol = duplicate_tol  # Structure similarity threshold
        self.fmax = fmax  # Optimization convergence criterion, max force threshold (eV/Å)
        self.max_steps = max_steps  # Maximum algorithm iterations
        self.output_dir = output_dir  # Output directory
        self.H = H  # Maximum number of Gaussian potentials
        self.w = w  # Gaussian potential height
        self.temperature = temperature  # Temperature parameter for Metropolis criterion
        self.k_boltzmann = 8.617333262e-5  # Boltzmann constant (eV/K)
        
        # New parameter assignments
        self.radius = radius  # Core region radius
        self.decay = decay  # Decay coefficient
        self.wall_strength = wall_strength  # Wall potential strength
        self.wall_offset = wall_offset  # Wall potential distance offset
        
        # Initialize mobility control parameters - New simplified design
        self.mobility_control = mobility_control
        
        if mobility_control is None:
            # Default: all atoms are mobile
            self.mobile_atoms = None  # None means all atoms mobile
            self.mobility_region = None
            self.wall_strength = 0.0  # No wall potential by default
            self.wall_offset = 2.0
        elif isinstance(mobility_control, dict):
            # Parse mobility control configuration
            mode = mobility_control.get('mode', 'all')  # 'all', 'region', 'indices_free', or 'indices_fix'
            
            if mode == 'all':
                # All atoms mobile (default)
                self.mobile_atoms = None
                self.mobility_region = None
            elif mode == 'indices_free':
                # Specific atoms explicitly set as mobile; others are fixed
                idx = mobility_control.get('indices_free', [])
                self.mobile_atoms = np.array(idx, dtype=int)
                self.mobility_region = None
            elif mode == 'indices_fix':
                # Specific atoms explicitly set as fixed; others are mobile
                fixed = np.array(mobility_control.get('indices_fix', []), dtype=int)
                all_idx = np.arange(len(self.atoms), dtype=int)
                mobile = np.setdiff1d(all_idx, fixed, assume_unique=False)
                self.mobile_atoms = mobile
                self.mobility_region = None
            elif mode == 'region':
                # Region-based control
                self.mobile_atoms = None
                region_type = mobility_control.get('region_type', 'sphere')
                
                if region_type == 'sphere':
                    # Sphere region: center + radius
                    self.mobility_region = {
                        'type': 'sphere',
                        'center': np.array(mobility_control.get('center', [0, 0, 0])),
                        'radius': mobility_control.get('radius', 10.0)
                    }
                elif region_type == 'slab':
                    # Slab region: between two planes
                    # Planes defined by normal and two distances
                    self.mobility_region = {
                        'type': 'slab',
                        'normal': np.array(mobility_control.get('normal', [0, 0, 1])),
                        'min_dist': mobility_control.get('min_dist', -5.0),
                        'max_dist': mobility_control.get('max_dist', 5.0),
                        'origin': np.array(mobility_control.get('origin', [0, 0, 0]))
                    }
                else:
                    raise ValueError(f"Unknown region_type: {region_type}")
            else:
                raise ValueError(f"Unknown mobility control mode: {mode}")
            
            # Wall potential settings
            self.wall_strength = mobility_control.get('wall_strength', wall_strength)
            self.wall_offset = mobility_control.get('wall_offset', wall_offset)
        else:
            raise ValueError(f"Invalid mobility_control type: {type(mobility_control)}")
        
        # Handle additional parameters
        self.additional_params = kwargs
        
        os.makedirs(output_dir, exist_ok=True)
        self.pool = []  # Stores all found minimum energy structures
        self.real_energies = []  # Stores corresponding structure energies
        
        # Initial structure optimization (real potential)
        self.atoms.calc = self.base_calc
        self._local_minimize(self.atoms)
        
        # Detect geometry type using occupancy ratios (cluster/slab/wire/bulk)
        self.geometry_type, self.vacuum_axes = self._infer_geometry_type(self.atoms)
        self.is_cluster = (self.geometry_type == 'cluster')
        # Axes with significant vacuum margins (set in _infer_geometry_type)
        # Compute box center in Cartesian
        cell = self.atoms.get_cell()
        box_center = (cell[0] + cell[1] + cell[2]) / 2.0
        # Determine initial center and optionally pre-center along vacuum axes
        pos0 = self.atoms.get_positions()
        curr_center = pos0.mean(axis=0)
        self.initial_cluster_center = None
        if len(self.vacuum_axes) > 0:
            # Shift only along vacuum axes so center moves to box_center
            shift = box_center - curr_center
            for i in range(3):
                if i not in self.vacuum_axes:
                    shift[i] = 0.0
            self.atoms.set_positions(pos0 + shift)
            # Use box center as the reference center for subsequent recentering
            self.initial_cluster_center = box_center
        axis_labels = ['x', 'y', 'z']
        dims = [axis_labels[i] for i in self.vacuum_axes]
        print(f"Detected geometry: {self.geometry_type}. Cluster: {self.is_cluster}. Vacuum axes: {self.vacuum_axes} ({', '.join(dims) if dims else 'none'})")
        if self.initial_cluster_center is not None:
            print(f"Initial center set to box center on axes {', '.join(dims)}: {self.initial_cluster_center}")
            print("Recentering will be applied after accepted moves along the vacuum axes.")
        self._add_to_pool(self.atoms)

    def _local_minimize(self, atoms, calc=None, fmax=None):
        """
        Perform local structure optimization using LBFGS algorithm
        Complies with CoSMoS algorithm documentation steps 4 and 6, using limited-memory BFGS optimizer for efficiency
        
        Parameters:
            atoms: Atomic structure to optimize
            calc: Calculator for optimization, defaults to base_calc
            fmax: Convergence force threshold, defaults to class instance's fmax
        """
        if calc is not None:
            atoms.calc = calc
        # Use LBFGS optimizer instead of BFGS, complies with documentation steps 4 and 6
        from ase.optimize import LBFGS
        opt = LBFGS(atoms, logfile=None)
        opt.run(fmax=fmax or self.fmax)

    def _add_to_pool(self, atoms):
        """
        Add optimized structure to minima pool and save to file
        """
        # Ensure calculator is attached before getting energy
        if atoms.calc is None:
            atoms.calc = self.base_calc
        real_e = atoms.get_potential_energy()
        self.pool.append(atoms.copy())
        self.real_energies.append(real_e)
        idx = len(self.pool) - 1
        
        # Append to the combined trajectory file instead of individual files
        atoms_copy = atoms.copy()
        atoms_copy.info['energy'] = real_e
        atoms_copy.info['minima_index'] = idx
        
        # Append mode: add structure to existing file
        from ase.io import write as ase_write
        trajectory_file = os.path.join(self.output_dir, 'all_minima.xyz')
        ase_write(trajectory_file, atoms_copy, append=True)
        
        print(f"Found new minimum #{idx}: E = {real_e:.6f} eV")

    def _get_mobility_mask(self):
        """
        Get boolean mask for mobile atoms.
        Returns array where True = mobile, False = immobile
        """
        n_atoms = len(self.atoms)
        
        # Default: all atoms mobile
        if self.mobile_atoms is None and self.mobility_region is None:
            return np.ones(n_atoms, dtype=bool)
        
        # Index-based control
        if self.mobile_atoms is not None:
            mask = np.zeros(n_atoms, dtype=bool)
            mask[self.mobile_atoms] = True
            return mask
        
        # Region-based control
        if self.mobility_region is not None:
            positions = self.atoms.get_positions()
            mask = np.zeros(n_atoms, dtype=bool)
            
            if self.mobility_region['type'] == 'sphere':
                center = self.mobility_region['center']
                radius = self.mobility_region['radius']
                distances = np.linalg.norm(positions - center, axis=1)
                mask = distances <= radius
                
            elif self.mobility_region['type'] == 'slab':
                normal = self.mobility_region['normal']
                normal = normal / np.linalg.norm(normal)
                origin = self.mobility_region['origin']
                min_dist = self.mobility_region['min_dist']
                max_dist = self.mobility_region['max_dist']
                
                # Project positions onto normal direction
                vectors = positions - origin
                distances = np.dot(vectors, normal)
                mask = (distances >= min_dist) & (distances <= max_dist)
            
            return mask
        
        return np.ones(n_atoms, dtype=bool)
    
    def _calculate_wall_potential(self, atoms):
        """
        Calculate wall potential to keep mobile atoms from entering too deep into immobile regions.
        Mobile atoms can penetrate wall_offset distance into immobile region before feeling repulsion.
        
        Returns:
            wall_energy: Total wall potential energy
            wall_forces: Forces from wall potential (3N flattened array)
        """
        if self.wall_strength == 0 or self.mobility_region is None:
            return 0.0, np.zeros(3 * len(atoms))
        
        positions = atoms.get_positions()
        n_atoms = len(atoms)
        wall_energy = 0.0
        wall_forces = np.zeros(3 * n_atoms)
        
        mobile_mask = self._get_mobility_mask()
        
        if self.mobility_region['type'] == 'sphere':
            # For sphere: push atoms back if they go too far from center
            center = self.mobility_region['center']
            radius = self.mobility_region['radius']
            
            for i in range(n_atoms):
                if not mobile_mask[i]:
                    continue
                
                delta_r = positions[i] - center
                dist = np.linalg.norm(delta_r)
                
                # Allow penetration up to wall_offset beyond the mobile boundary
                # If atom goes beyond (radius + wall_offset), apply repulsive force
                if dist > radius + self.wall_offset:
                    # Distance beyond allowed penetration
                    overshoot = dist - radius - self.wall_offset
                    # Quadratic potential: V = 0.5 * k * overshoot^2
                    wall_energy += 0.5 * self.wall_strength * overshoot ** 2
                    # Force: F = -k * overshoot * (direction)
                    direction = delta_r / dist if dist > 0 else np.zeros(3)
                    force = -self.wall_strength * overshoot * direction
                    wall_forces[3*i:3*i+3] = force
        
        elif self.mobility_region['type'] == 'slab':
            # For slab: push atoms back if they penetrate too far through the planes
            normal = self.mobility_region['normal']
            normal = normal / np.linalg.norm(normal)
            origin = self.mobility_region['origin']
            min_dist = self.mobility_region['min_dist']
            max_dist = self.mobility_region['max_dist']
            
            for i in range(n_atoms):
                if not mobile_mask[i]:
                    continue
                
                # Project position onto normal direction
                delta_r = positions[i] - origin
                proj_dist = np.dot(delta_r, normal)
                
                # Check penetration beyond allowed boundaries
                if proj_dist < min_dist - self.wall_offset:
                    # Penetrated too far through lower plane
                    overshoot = (min_dist - self.wall_offset) - proj_dist
                    wall_energy += 0.5 * self.wall_strength * overshoot ** 2
                    # Force pushes in +normal direction
                    force = self.wall_strength * overshoot * normal
                    wall_forces[3*i:3*i+3] = force
                    
                elif proj_dist > max_dist + self.wall_offset:
                    # Penetrated too far through upper plane
                    overshoot = proj_dist - (max_dist + self.wall_offset)
                    wall_energy += 0.5 * self.wall_strength * overshoot ** 2
                    # Force pushes in -normal direction
                    force = -self.wall_strength * overshoot * normal
                    wall_forces[3*i:3*i+3] = force
        
        return wall_energy, wall_forces

    def _get_atomic_energies(self, atoms):
        """
        Get per-atom energies. If the calculator doesn't support it (e.g., VASP),
        use a fallback FAIRChem calculator.
        
        Returns:
            np.ndarray: Per-atom energies
        """
        temp_atoms = atoms.copy()
        temp_atoms.calc = self.base_calc
        
        try:
            # Try to get per-atom energies from the calculator
            atomic_energies = temp_atoms.get_potential_energies()
            return atomic_energies
        except (AttributeError, NotImplementedError, RuntimeError) as e:
            # Calculator doesn't support per-atom energies, use FAIRChem fallback
            print(f"Warning: Base calculator doesn't support per-atom energies. Using FAIRChem fallback.")
            try:
                # Import and setup FAIRChem calculator if not already cached
                if not hasattr(self, '_fairchem_calc'):
                    from fairchem.core import OCPCalculator
                    # Use a lightweight pretrained model for fast inference
                    self._fairchem_calc = OCPCalculator(
                        model_name="EquiformerV2-31M-S2EF-OC20-All+MD",
                        cpu=True
                    )
                    print("FAIRChem calculator initialized for per-atom energy calculation.")
                
                # Calculate per-atom energies with FAIRChem
                temp_atoms_fc = atoms.copy()
                temp_atoms_fc.calc = self._fairchem_calc
                atomic_energies = temp_atoms_fc.get_potential_energies()
                return atomic_energies
            except Exception as fc_error:
                print(f"Warning: FAIRChem fallback also failed: {fc_error}")
                print("Falling back to uniform distribution for random direction generation.")
                # Return uniform energies as last resort
                return np.zeros(len(atoms))
    
    def _get_energy_based_scales(self, atoms):
        """
        Calculate energy-based scales for each atom to guide random direction generation.
        Atoms with higher energy (less stable) get larger scales for Ns components.
        
        Returns:
            np.ndarray: Per-atom scales based on exp(normalized_energy)
        """
        atomic_energies = self._get_atomic_energies(atoms)
        
        # Get chemical symbols and find minimum energy for each element as reference
        symbols = atoms.get_chemical_symbols()
        unique_elements = list(set(symbols))
        
        # Calculate reference energies (minimum energy for each element)
        reference_energies = {}
        for element in unique_elements:
            element_indices = [i for i, sym in enumerate(symbols) if sym == element]
            element_energies = atomic_energies[element_indices]
            reference_energies[element] = np.min(element_energies)
        
        # Normalize energies relative to element-specific references
        normalized_energies = np.zeros(len(atoms))
        for i, symbol in enumerate(symbols):
            normalized_energies[i] = atomic_energies[i] - reference_energies[symbol]
        
        # Ensure all normalized energies are non-negative (they should be by construction)
        normalized_energies = np.maximum(normalized_energies, 0.0)
        
        # Calculate scales using exp(E_atom)
        # Add small offset to avoid issues with very small energies
        scales = np.exp(normalized_energies)
        
        # Normalize scales to have reasonable magnitude (mean = 1)
        mean_scale = np.mean(scales)
        if mean_scale > 0:
            scales = scales / mean_scale
        
        return scales

    def _generate_random_direction(self, atoms):
        """
        Generate random search direction, combining global soft movement and local rigid movement.
        Uses energy-based sampling: atoms with higher energy get larger random components.
        Complies with CoSMoS algorithm step 1: Generate initial random direction N⁰
        
        Returns:
            N: Normalized random direction vector
        """
        n_atoms = len(atoms)
        
        # Get energy-based scales for each atom
        # Atoms with higher energy (less stable) will have larger scales
        energy_scales = self._get_energy_based_scales(atoms)
        
        # Generate global soft movement direction Ns using energy-weighted Gaussian sampling
        # Instead of uniform Maxwell-Boltzmann, we use exp(E_atom) as scale for each atom
        Ns = np.zeros(3 * n_atoms)
        for i in range(n_atoms):
            # Each atom gets a Gaussian random vector with scale proportional to exp(E_atom)
            # Base scale from temperature
            mass = 1.0  # Atomic mass unit
            #base_scale = np.sqrt(self.k_boltzmann * self.temperature / mass)
            base_scale = np.sqrt(self.k_boltzmann * self.temperature)
            # Multiply by energy-based scale for this atom
            atom_scale = base_scale * energy_scales[i]
            # Generate 3 random components for this atom
            Ns[3*i:3*i+3] = np.random.normal(0, atom_scale, 3)
        
        # Apply mobility mask: set Ns to zero for immobile atoms
        if self.mobility_control is not None:
            # Get mobility mask (True = mobile, False = immobile)
            mobile_mask = self._get_mobility_mask()
            # Set Ns components to zero for immobile atoms
            for i in range(n_atoms):
                if not mobile_mask[i]:
                    Ns[3*i:3*i+3] = 0.0
        
        # Generate local rigid movement direction Nl (non-adjacent atom bonding pattern)
        Nl = np.zeros(3 * n_atoms)
        if n_atoms >= 2:
            # Randomly select two non-adjacent atoms
            # Filter atoms in mobile region
            mobile_mask = self._get_mobility_mask()
            mobile_atoms = np.where(mobile_mask)[0].tolist()
            
            # Check if there are enough mobile atoms
            if len(mobile_atoms) < 2:
                raise ValueError("Insufficient mobile atoms (need at least 2) for calculation.")
            
            max_attempts = 50
            attempts = 0
            found = False
            
            while attempts < max_attempts and not found:
                # Select two non-neighboring atoms from mobile region
                indices = np.random.choice(mobile_atoms, 2, replace=False)
                i, j = indices
                qi = atoms.positions[i].flatten()
                qj = atoms.positions[j].flatten()
                distance = np.linalg.norm(qi - qj)
                
                if distance > 3.0:  # Only when atomic distance > 3Å
                    # Generate local rigid movement direction according to equation (2) in paper
                    Nl[3*i:3*i+3] = qj
                    Nl[3*j:3*j+3] = qi
                    norm = np.linalg.norm(Nl)
                    if norm == 0:
                        raise ValueError("Norm of Nl direction vector is zero, cannot normalize.")
                    Nl /= norm
                    found = True
                
                attempts += 1
            
            if not found:
                raise ValueError(f"Failed to find atom pair with distance > 3Å after {max_attempts} attempts, cannot generate local rigid movement direction.")
        
        # Mix direction vectors - according to paper equation (1)
        lambda_param = np.random.uniform(0.1, 1.5)
        # Ensure Ns and Nl are normalized
        Ns_norm = np.linalg.norm(Ns)
        Nl_norm = np.linalg.norm(Nl)
        
        if Ns_norm > 0:
            Ns = Ns / Ns_norm
        if Nl_norm > 0:
            Nl = Nl / Nl_norm
            
        N = Ns + lambda_param * Nl
        N /= np.linalg.norm(N) if np.linalg.norm(N) > 0 else 1
        return N

    def _apply_random_move(self, atoms, direction):
        """Apply random movement to atomic structure"""
        new_atoms = atoms.copy()
        # Apply mobility mask: only mobile atoms move
        if self.mobility_control is not None:
            mobile_mask = self._get_mobility_mask()
            for i in range(len(atoms)):
                if not mobile_mask[i]:
                    direction[3*i:3*i+3] = 0.0
        # Apply direction vector and step size
        new_pos = new_atoms.get_positions().flatten() + self.ds * direction
        new_atoms.set_positions(new_pos.reshape(-1, 3))
        return new_atoms

    def _get_real_energy(self, atoms):
        """
        Get energy of structure on real potential energy surface
        """
        temp_atoms = atoms.copy()
        temp_atoms.calc = self.base_calc
        return temp_atoms.get_potential_energy()
    
    def _translate_to_initial_center(self, atoms):
        """
        Translate structure so its geometric center returns to the initial center
        along axes that have significant vacuum margins (self.vacuum_axes).
        """
        if getattr(self, 'initial_cluster_center', None) is not None and len(self.vacuum_axes) > 0:
            pos = atoms.get_positions()
            curr_center = pos.mean(axis=0)
            shift = self.initial_cluster_center - curr_center
            # Zero out components for non-vacuum axes
            for i in range(3):
                if i not in self.vacuum_axes:
                    shift[i] = 0.0
            atoms.set_positions(pos + shift)
    
    def _infer_geometry_type(self, atoms, vacuum_threshold_angstrom: float = 3.0):
        """
        Infer geometry type based on absolute vacuum margins (Å) along each lattice axis.
        Returns a tuple: (geometry_type, vacuum_axes).
        vacuum_axes lists axes with significant vacuum on both sides.
        """
        # Use absolute positions for vacuum detection (avoid relying solely on fractional coords)
        pos = atoms.get_positions()  # Cartesian positions within the cell
        cell = atoms.get_cell()
        axis_lengths = np.array([np.linalg.norm(cell[0]), np.linalg.norm(cell[1]), np.linalg.norm(cell[2])])
        # Compute absolute margins to cell boundaries for each axis
        min_abs = pos.min(axis=0)
        max_abs = pos.max(axis=0)
        margin_low_abs = min_abs  # distance to 0 boundary
        margin_high_abs = axis_lengths - max_abs  # distance to upper boundary
        axes = []
        for i in range(3):
            if margin_low_abs[i] > vacuum_threshold_angstrom and margin_high_abs[i] > vacuum_threshold_angstrom:
                axes.append(i)
        n = len(axes)
        if n == 3:
            geom = 'cluster'
        elif n == 2:
            geom = 'wire'
        elif n == 1:
            geom = 'slab'
        else:
            geom = 'bulk'
        return geom, axes

    def _write_step_output(self, step, atoms, energy):
        """
        Output step information to file
        """
        with open(os.path.join(self.output_dir, 'cosmos_log.txt'), 'a') as f:
            f.write(f"Step {step+1}: Energy = {energy:.6f} eV\n")
    
    def run(self, steps=100):
        """
        Run CoSMoS global search algorithm, strictly following steps in the paper
        
        Parameters:
            steps: Total algorithm iterations
        """
        # Initialize log file
        header_lines = []
        try:
            with open(os.path.join(self.output_dir, 'cosmos_log.txt'), 'r') as f:
                header_lines = [next(f) for _ in range(5)]  # Read first 5 lines (header)
        except Exception:
            header_lines = ["CoSMoS Search Log\n"]
        
        with open(os.path.join(self.output_dir, 'cosmos_log.txt'), 'a') as f:
            f.write(f"Initial structure: Energy = {self._get_real_energy(self.atoms):.6f} eV\n")
        
        # Initialize combined trajectory file (remove old one if exists)
        trajectory_file = os.path.join(self.output_dir, 'all_minima.xyz')
        if os.path.exists(trajectory_file):
            os.remove(trajectory_file)
        
        # Initialize current structure as initial minimum structure
        current_atoms = self.atoms.copy()
        current_energy = self._get_real_energy(current_atoms)
        
        for step in range(steps):
            print(f"\n--- CoSMoS Step {step + 1}/{steps} ---")
            
            # Algorithm Step 1: Generate initial random direction N⁰ at current minimum Rm
            direction = self._generate_random_direction(current_atoms)
            
            # Algorithm Step 2: Optimize direction using biased dimer rotation to get N¹
            optimized_direction = self._biased_dimer_rotation(current_atoms, direction)
            
            # Algorithm Steps 3-4: Climbing phase - add Gaussian potentials and locally optimize
            climb_atoms = current_atoms.copy()
            gaussian_params = []
            Emax = current_energy  # Record highest energy during climbing phase
            
            for n in range(1, self.H + 1):
                # Calculate direction vector
                if n == 1:
                    # First step uses optimized direction
                    N = optimized_direction
                else:
                    # Subsequent steps generate new random directions
                    new_rand_dir = self._generate_random_direction(climb_atoms)
                    # Optimize new direction using biased dimer rotation
                    N = self._biased_dimer_rotation(climb_atoms, new_rand_dir)
                
                # Apply mobility mask
                if self.mobility_control is not None:
                    mobile_mask = self._get_mobility_mask()
                    for i in range(len(climb_atoms)):
                        if not mobile_mask[i]:
                            N[3*i:3*i+3] = 0.0
                    N /= np.linalg.norm(N) if np.linalg.norm(N) > 0 else 1
                
                # CRITICAL: Move structure along direction N before adding bias potential
                # This is Step 3 in the SSW paper: R^{n-1} displacement by ds along N_i^n
                climb_atoms_positions = climb_atoms.get_positions().flatten()
                climb_atoms_positions += self.ds * N
                climb_atoms.set_positions(climb_atoms_positions.reshape(-1, 3))
                
                # Add new Gaussian potential at the displaced position
                g_param = self._add_gaussian(climb_atoms, N)
                gaussian_params.append(g_param)
                
                # Calculate wall potential for current configuration
                wall_energy, wall_forces = self._calculate_wall_potential(climb_atoms)
                
                # Create biased calculator
                biased_calc = BiasedCalculator(
                    base_calculator=self.base_calc,
                    gaussian_params=gaussian_params,
                    ds=self.ds,
                    wall_energy=wall_energy,
                    wall_forces=wall_forces
                )
                
                # Locally optimize on modified potential energy surface
                climb_atoms.calc = biased_calc
                self._local_minimize(climb_atoms)
                
                # Compute real energy
                current_climb_energy = self._get_real_energy(climb_atoms)
                Emax = max(Emax, current_climb_energy)
                
                # Algorithm Step 5: Check stopping condition
                # Stop if: (i) reached max Gaussians H, or (ii) structure relaxed back below starting energy
                if n >= self.H:
                    print(f"Climb phase end: n={n}, reached maximum Gaussians")
                    break
                if current_climb_energy <= current_energy:
                    print(f"Climb phase end: n={n}, energy {current_climb_energy:.6f} eV <= initial {current_energy:.6f} eV")
                    break
            
            # Algorithm Step 6: Remove all biased potentials and optimize on real potential energy surface
            climb_atoms.calc = self.base_calc
            self._local_minimize(climb_atoms)
            relaxed_energy = self._get_real_energy(climb_atoms)
            
            # Algorithm Step 7: Use Metropolis criterion to accept or reject
            delta_E = relaxed_energy - current_energy
            if delta_E > 0:
                accept_prob = np.exp(-delta_E / (self.k_boltzmann * self.temperature))
            else:
                accept_prob = 1.0
            
            if np.random.rand() < accept_prob:
                print(f"Accept new structure: ΔE = {delta_E:.6f} eV, P = {accept_prob:.4f}")
                # Check if new structure is a duplicate
                if not is_duplicate_by_desc_and_energy(climb_atoms, self.pool, self.soap_species, tol=self.duplicate_tol, energy=relaxed_energy, pool_energies=self.real_energies, energy_tol=1):
                    # New unique structure found
                    # Align cluster center to initial center to prevent global translation
                    self._translate_to_initial_center(climb_atoms)
                    current_atoms = climb_atoms.copy()
                    current_atoms.calc = self.base_calc  # Attach calculator
                    current_energy = relaxed_energy
                    self._add_to_pool(current_atoms)
                else:
                    # Duplicate structure, but still update current_atoms to explore from different point
                    # This prevents getting stuck in the same location
                    print("Structure is duplicate, not added to pool")
                    # Randomly perturb to escape local minimum
                    if delta_E == 0.0:  # Exactly same energy means stuck
                        print("Warning: Stuck at same structure, applying random perturbation")
                        perturb_direction = self._generate_random_direction(climb_atoms)
                        perturb_positions = climb_atoms.get_positions().flatten()
                        perturb_positions += 0.1 * self.ds * perturb_direction  # Small perturbation
                        climb_atoms.set_positions(perturb_positions.reshape(-1, 3))
                        climb_atoms.calc = self.base_calc
                        self._local_minimize(climb_atoms)
                    self._translate_to_initial_center(climb_atoms)
                    current_atoms = climb_atoms.copy()
                    current_atoms.calc = self.base_calc  # Attach calculator
                    current_energy = self._get_real_energy(current_atoms)
            else:
                print(f"Reject new structure: ΔE = {delta_E:.6f} eV, P = {accept_prob:.4f}")
            
            # Output current step information
            self._write_step_output(step, current_atoms, current_energy)
        
        print("\nCoSMoS search completed!")
        print(f"All {len(self.pool)} minima structures saved to: {os.path.join(self.output_dir, 'all_minima.xyz')}")
        
        # Save the lowest energy structure
        if self.pool and self.real_energies:
            min_idx = self.real_energies.index(min(self.real_energies))
            best_atoms = self.pool[min_idx].copy()
            best_energy = self.real_energies[min_idx]
            best_atoms.info['energy'] = best_energy
            best_atoms.info['minima_index'] = min_idx
            
            from ase.io import write as ase_write
            best_file = os.path.join(self.output_dir, 'best_str.xyz')
            ase_write(best_file, best_atoms)
            print(f"Lowest energy structure (E = {best_energy:.6f} eV) saved to: {best_file}")
        
        return self.pool, self.real_energies

    def _biased_dimer_rotation(self, atoms, initial_direction):
        """
        Implement biased dimer rotation method according to SSW paper (Eq. 3-6)
        Uses proper dimer method to find the lowest curvature direction with bias potential
        
        Reference: Shang & Liu, J. Chem. Phys. 139, 244104 (2013)
        
        Parameters:
            atoms: Current atomic structure (at minimum R_m)
            initial_direction: Initial search direction N^0
        
        Returns:
            Optimized direction N^1 (normalized)
        """
        # Normalize initial direction
        N = initial_direction.copy()
        norm_N = np.linalg.norm(N)
        if norm_N > 0:
            N = N / norm_N
        else:
            return initial_direction
        
        # Dimer parameters from SSW paper
        delta_R = 0.005  # Dimer separation (typical: 0.005 Å)
        theta_trial = 0.5 * np.pi / 180.0  # Trial rotation angle (radians), ~0.5 degrees
        max_rotations = 10  # Maximum number of rotations
        f_rot_tol = 0.01  # Rotational force tolerance (eV/Å)
        
        # Bias potential parameter (from Eq. 6 in paper)
        a = 1.0 / (self.ds ** 2)  # Parameter controlling bias potential strength
        
        # Current position (flattened)
        R0_flat = atoms.positions.flatten()
        
        # Iteratively rotate dimer to find optimal direction
        for rotation_iter in range(max_rotations):
            # Calculate dimer images: R_1 = R_0 + N * ΔR (Eq. 3)
            R1_flat = R0_flat + N * delta_R
            
            # Compute forces at R_1 (on real PES)
            temp_atoms1 = atoms.copy()
            temp_atoms1.set_positions(R1_flat.reshape(-1, 3))
            temp_atoms1.calc = self.base_calc
            F1 = temp_atoms1.get_forces().flatten()
            
            # Compute curvature C = (F_0 - F_1) · N / ΔR (Eq. 4)
            # Note: F_0 can be extrapolated from F_1 for efficiency
            # F_0 ≈ -F_1 (symmetric approximation for small ΔR)
            F0_approx = -F1
            C = np.dot((F0_approx - F1), N) / delta_R
            
            # Calculate rotational force (perpendicular component)
            # F_rot = F_1 - (F_1 · N) * N  (force perpendicular to N)
            F1_parallel = np.dot(F1, N) * N
            F_rot = F1 - F1_parallel
            
            # Check convergence: if rotational force is small, stop rotation
            F_rot_mag = np.linalg.norm(F_rot)
            if F_rot_mag < f_rot_tol:
                break
            
            # Compute rotation angle using finite difference
            # Trial rotation: N' = N * cos(θ) + F_rot_normalized * sin(θ)
            if F_rot_mag > 1e-10:
                F_rot_normalized = F_rot / F_rot_mag
            else:
                break  # No significant rotation needed
            
            # Trial rotation with small angle
            N_trial = N * np.cos(theta_trial) + F_rot_normalized * np.sin(theta_trial)
            N_trial = N_trial / np.linalg.norm(N_trial)
            
            # Evaluate curvature at trial position
            R1_trial_flat = R0_flat + N_trial * delta_R
            temp_atoms_trial = atoms.copy()
            temp_atoms_trial.set_positions(R1_trial_flat.reshape(-1, 3))
            temp_atoms_trial.calc = self.base_calc
            F1_trial = temp_atoms_trial.get_forces().flatten()
            F0_trial_approx = -F1_trial
            C_trial = np.dot((F0_trial_approx - F1_trial), N_trial) / delta_R
            
            # Estimate optimal rotation angle using finite difference
            # dC/dθ ≈ (C_trial - C) / θ_trial
            dC_dtheta = (C_trial - C) / theta_trial
            
            # Compute second derivative estimate for parabolic fit
            # d²C/dθ² ≈ 2 * C / θ²  (approximate)
            if abs(theta_trial) > 1e-10:
                # Optimal angle from parabolic approximation: θ_opt = -dC/dθ / (d²C/dθ²)
                # Use simple formula: θ_opt = θ_trial * C / (C - C_trial)
                if abs(C - C_trial) > 1e-10:
                    theta_opt = theta_trial * C / (C - C_trial)
                    # Limit rotation angle to avoid overshooting
                    theta_opt = np.clip(theta_opt, -np.pi/4, np.pi/4)
                else:
                    theta_opt = 0.0
            else:
                theta_opt = 0.0
            
            # Apply optimal rotation
            if abs(theta_opt) > 1e-6:
                N = N * np.cos(theta_opt) + F_rot_normalized * np.sin(theta_opt)
                N = N / np.linalg.norm(N)
            else:
                break  # Converged
        
        return N



    def _add_gaussian(self, atoms, direction):
        """
        Generate new Gaussian potential parameters, according to the definition in the paper
        
        Parameters:
            atoms: Current atomic structure
            direction: Gaussian potential direction vector
            
        Returns:
            tuple: (d, R1, w) - contains unit direction vector, reference position and height parameters of the tuple
        """
        # Ensure direction vector is unit vector
        norm = np.linalg.norm(direction)
        d = direction / norm if norm > 0 else direction
        
        # Use current atomic position as reference position R1
        R1 = atoms.positions.copy()
        
        # Adjust Gaussian potential height according to the climbing phase, earlier steps use smaller w, later steps use larger w to encourage further exploration
        w = self.w  # Can be adjusted according to strategy
        
        return (d, R1, w)
    
    def get_minima_pool(self):
        """
        Get all found minima structures pool
        
        Returns:
            list: Contains all found minima Atoms structures list
        """
        return self.pool