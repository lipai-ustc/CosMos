# cosmos_search.py (v2: faithful to Shang & Liu 2013)
# Reference 1: Shang, R., & Liu, J. (2013). Stochastic surface walking method for global optimization of atomic clusters and biomolecules. The Journal of Chemical Physics, 139(24), 244104.
# Reference 2: J. Chem. Theory Comput. 2012, 8, 2215
import os
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.calculators.calculator import Calculator, all_changes
from cosmos_utils import is_duplicate_by_desc_and_energy, infer_geometry_type, get_mobility_mask, calculate_wall_potential

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
            
            # Gaussian width uses self.ds (step size); equation parameters consolidated
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
        self.results['energy_components'] = (E0, V_bias_total, self.wall_energy)


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
        random_direction_mode="atomic_plus_nl",
        mobility_control=None,            # Mobility control configuration dict or None
        potential_type=None,              # Potential type ('deepmd', 'eam', etc.)
        nequip_config=None,               # NequIP configuration for fallback atomic energy
        debug=False,                      # Debug mode for detailed logging
        **kwargs                      # Additional parameters
    ):
        # Initialize minimum pool
        self.pool = []  # Stores found minimum energy structures
        self.atoms = initial_atoms.copy()
        self.base_calc = calculator  # Calculator for real potential energy
        self.potential_type = potential_type  # Store potential type for calculator selection
        self.nequip_config = nequip_config  # Store NequIP config for atomic energy fallback
        self.debug = debug  # Debug mode flag
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
        self.random_direction_mode = random_direction_mode.strip().lower()
        
        # New parameter assignments
        self.radius = radius  # Core region radius
        self.decay = decay  # Decay coefficient
        
        # Mobility control is fully normalized in cosmos_run.py; do not parse input.json here
        self.mobile_atoms = None
        self.mobility_region = None
        # Default wall settings
        self.wall_strength = wall_strength
        self.wall_offset = wall_offset
        if isinstance(mobility_control, dict):
            self.mobile_atoms = mobility_control['mobile_atoms']
            self.mobility_region = mobility_control['mobility_region']
            self.wall_strength = mobility_control['wall_strength']
            self.wall_offset = mobility_control['wall_offset']
        elif mobility_control is not None:
            raise ValueError(f"Invalid mobility_control type: {type(mobility_control)}")
        
        # Handle additional parameters
        self.additional_params = kwargs
        
        os.makedirs(output_dir, exist_ok=True)
        self.pool = []  # Stores all found minimum energy structures
        self.real_energies = []  # Stores corresponding structure energies
        
        # Compute mobility mask once at initialization
        # mobility_region is in fractional coordinates and does not change during SSW
        if (self.mobile_atoms is not None) or (self.mobility_region is not None):
            self.mobile_mask = get_mobility_mask(self.atoms, self.mobile_atoms, self.mobility_region)
            # Cache number of mobile atoms
            self.n_mobile = int(np.sum(self.mobile_mask))
            # Apply FixAtoms constraint to immobile atoms
            # This ensures local minimization respects mobility constraints
            fixed_indices = [i for i in range(len(self.atoms)) if not self.mobile_mask[i]]
            if len(fixed_indices) > 0:
                constraint = FixAtoms(indices=fixed_indices)
                self.atoms.set_constraint(constraint)
                print(f"Applied FixAtoms constraint to {len(fixed_indices)} immobile atoms.")
        else:
            self.mobile_mask = None
            # Cache when all atoms are mobile
            self.n_mobile = len(self.atoms)
        
        # Initial structure optimization (real potential)
        self.atoms.calc = self.base_calc
        self._local_minimize(self.atoms)
        
        # Detect geometry type using occupancy ratios (cluster/slab/wire/bulk)
        self.geometry_type, self.vacuum_axes = infer_geometry_type(self.atoms)
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
        
        # Use LBFGS optimizer with custom logging if debug mode
        from ase.optimize import LBFGS
        
        if self.debug and isinstance(calc, BiasedCalculator):
            # Debug mode: log energy components at each step
            import sys
            
            class DebugLBFGS(LBFGS):
                def __init__(self, atoms, parent_search, **kwargs):
                    super().__init__(atoms, **kwargs)
                    self.parent_search = parent_search
                    self.step_count = 0
                
                def step(self, f=None):
                    result = super().step(f)
                    self.step_count += 1
                    
                    # Log energy components
                    if hasattr(self.atoms.calc, 'results') and 'energy_components' in self.atoms.calc.results:
                        E_base, E_bias, E_wall = self.atoms.calc.results['energy_components']
                        E_total = self.atoms.get_potential_energy()
                        forces = self.atoms.get_forces()
                        fmax_current = (forces**2).sum(axis=1).max()**0.5
                        
                        print(f"  Step {self.step_count}: E_total = {E_total:.6f} eV, "
                              f"E_base = {E_base:.6f} eV, E_bias = {E_bias:.6f} eV, E_wall = {E_wall:.6f} eV, "
                              f"fmax = {fmax_current:.6f} eV/Å")
                    
                    return result
            
            opt = DebugLBFGS(atoms, self, logfile=None)
        else:
            # Normal mode: no step-by-step logging
            opt = LBFGS(atoms, logfile=None)
        
        opt.run(fmax=fmax or self.fmax)
        
        # Output final optimized energy
        final_energy = atoms.get_potential_energy()
        print(f"Local minimization completed: E = {final_energy:.6f} eV")
        
        # Output final energy components if available
        if isinstance(calc, BiasedCalculator) and hasattr(calc, 'results') and 'energy_components' in calc.results:
            E_base, E_bias, E_wall = calc.results['energy_components']
            print(f"  Final energy components: base = {E_base:.6f} eV, bias = {E_bias:.6f} eV, wall = {E_wall:.6f} eV")

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


    def _get_atomic_energies(self, atoms):
        """
        Get per-atom energies with fallback chain:
        1. Try primary calculator (native support)
        2. If deepmd: use DeepMDCalculatorWithAtomicEnergy
        3. If nequip: use NequIP's native per-atom energies
        4. If fails: try NequIP as fallback calculator
        5. Last resort: uniform distribution (zeros)
        
        Returns:
            np.ndarray: Per-atom energies
        """
        # Special handling for DeepMD: use custom wrapper
        if self.potential_type == 'deepmd':
            if not hasattr(self, '_deepmd_atomic_calc'):
                from cosmos_utils import DeepMDCalculatorWithAtomicEnergy
                model_path = self.base_calc.dp.model_file
                self._deepmd_atomic_calc = DeepMDCalculatorWithAtomicEnergy(model=model_path)
                print("Initialized DeepMDCalculatorWithAtomicEnergy for per-atom energy calculation.")
            
            temp_atoms = atoms.copy()
            temp_atoms.calc = self._deepmd_atomic_calc
            atomic_energies = temp_atoms.get_potential_energies()
            return atomic_energies
        
        # Try primary calculator first (including NequIP)
        temp_atoms = atoms.copy()
        temp_atoms.calc = self.base_calc
        
        try:
            atomic_energies = temp_atoms.get_potential_energies()
            return atomic_energies
        except (AttributeError, NotImplementedError, RuntimeError) as e:
            # Primary calculator doesn't support per-atom energies
            if not hasattr(self, '_atomic_energy_warning_shown'):
                print(f"Warning: Primary calculator ({self.potential_type}) doesn't support per-atom energies.")
                print("Attempting NequIP fallback for atomic energy calculation...")
                self._atomic_energy_warning_shown = True
            
            # Try NequIP as fallback calculator
            try:
                if not hasattr(self, '_nequip_atomic_calc'):
                    # Check if nequip_config is available in input
                    if hasattr(self, 'nequip_config') and self.nequip_config:
                        from nequip.ase import NequIPCalculator
                        model_path = self.nequip_config.get('model')
                        device = self.nequip_config.get('device', 'cpu')
                        self._nequip_atomic_calc = NequIPCalculator.from_compiled_model(
                            compile_path=model_path,
                            device=device
                        )
                        print(f"Initialized NequIP calculator from config for per-atom energy calculation.")
                    else:
                        # No NequIP config available, skip to uniform fallback
                        raise AttributeError("No NequIP configuration available for fallback.")
                
                temp_atoms_nq = atoms.copy()
                temp_atoms_nq.calc = self._nequip_atomic_calc
                atomic_energies = temp_atoms_nq.get_potential_energies()
                print("Successfully obtained per-atom energies from NequIP fallback.")
                return atomic_energies
                
            except Exception as nq_error:
                if not hasattr(self, '_nequip_fallback_warning_shown'):
                    print(f"Warning: NequIP fallback also failed: {nq_error}")
                    print("Using uniform energy distribution for random direction generation.")
                    self._nequip_fallback_warning_shown = True
                
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
        # Only consider mobile atoms for energy comparison
        symbols = atoms.get_chemical_symbols()
        unique_elements = list(set(symbols))
        
        # Determine which atoms are mobile
        if self.mobile_mask is not None:
            mobile_indices = [i for i in range(len(atoms)) if self.mobile_mask[i]]
        else:
            mobile_indices = list(range(len(atoms)))
        
        # Calculate reference energies (minimum energy for each element among mobile atoms)
        reference_energies = {}
        for element in unique_elements:
            # Find mobile atoms of this element
            element_mobile_indices = [i for i in mobile_indices if symbols[i] == element]
            if len(element_mobile_indices) > 0:
                element_energies = atomic_energies[element_mobile_indices]
                reference_energies[element] = np.min(element_energies)
            else:
                # No mobile atoms of this element, use 0 as reference
                reference_energies[element] = 0.0
        
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
        # Only use mobile atoms for normalization
        if len(mobile_indices) > 0:
            mean_scale = np.mean(scales[mobile_indices])
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
        mode = getattr(self, 'random_direction_mode', 'atomic_plus_nl')
        
        # If mode is 'python', load and use user-defined function
        if mode == 'python':
            return self._generate_random_direction_python(atoms)
        
        # Otherwise, use built-in methods
        n_atoms = len(atoms)
        
        # Generate global soft movement direction Ns according to mode
        Ns = np.zeros(3 * n_atoms)
        mass = 1.0  # Atomic mass unit
        base_scale = np.sqrt(self.k_boltzmann * self.temperature)
        mode = getattr(self, 'random_direction_mode', 'atomic_plus_nl')
        use_atomic = ('atomic' in mode)
        include_nl = ('plus_nl' in mode)
        
        # Get energy scales ONCE before loop (if needed)
        if use_atomic:
            energy_scales = self._get_energy_based_scales(atoms)
        
        for i in range(n_atoms):
            if use_atomic:
                # Atoms with higher energy (less stable) will have larger scales
                atom_scale = base_scale * energy_scales[i]
            else:
                atom_scale = base_scale
            Ns[3*i:3*i+3] = np.random.normal(0, atom_scale, 3)
        
        # Apply mobility mask: set Ns to zero for immobile atoms BEFORE normalization
        # This ensures fixed atoms don't contribute to the random direction
        if self.mobile_mask is not None:
            for i in range(n_atoms):
                if not self.mobile_mask[i]:
                    Ns[3*i:3*i+3] = 0.0
        print('Ns', Ns)  # Debug: print Ns vector
        
        # Generate local rigid movement direction Nl (non-adjacent atom bonding pattern)
        Nl = np.zeros(3 * n_atoms)
        selected_pair = None  # Track the selected atom pair for Nl
        pair_distance = 0.0
        if include_nl and (n_atoms >= 2):
            # Randomly select two non-adjacent atoms
            # Filter atoms in mobile region
            if self.mobile_mask is not None:
                mobile_atoms = np.where(self.mobile_mask)[0].tolist()
            else:
                mobile_atoms = list(range(n_atoms))
            
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
                    # Nl = [qB - qA at position A, qA - qB at position B, 0, ...]
                    Nl[3*i:3*i+3] = qj - qi
                    Nl[3*j:3*j+3] = qi - qj
                    # Scale Nl to a fixed magnitude to avoid excessive displacement for distant pairs
                    nl_norm = np.linalg.norm(Nl)
                    if nl_norm > 0:
                        Nl = (Nl / nl_norm) * 2.0
                    selected_pair = (i, j)
                    pair_distance = distance
                    found = True
                
                attempts += 1
            
            if not found:
                raise ValueError(f"Failed to find atom pair with distance > 3Å after {max_attempts} attempts, cannot generate local rigid movement direction.")
        
        # Mix direction vectors - according to paper equation (1)
        # Normalization strategy: scale by sqrt(N_mobile) to balance between:
        # 1) Avoiding dilution in large systems (each atom gets reasonable displacement)
        # 2) Preserving relative displacement magnitudes from energy weighting
        lambda_param = np.random.uniform(0.1, 1.5)
        
        # Count mobile atoms (cached)
        n_mobile = getattr(self, 'n_mobile', n_atoms)
        
        # Combine Ns and optional Nl
        if include_nl:
            N = Ns + lambda_param * Nl
        else:
            N = Ns
        
        # Normalize and scale by sqrt(n_mobile)
        N_norm = np.linalg.norm(N)
        print('N before norm', N)  # Debug: print N vector        
        if N_norm > 0:
            N = N / N_norm * np.sqrt(n_mobile)
        print('N', N)  # Debug: print N vector
        # Debug output: show displacement statistics
        print(f"\n=== Random Direction Generation ===")
        print(f"Number of atoms: {n_atoms}, Mobile atoms: {n_mobile}")
        print(f"Lambda: {lambda_param:.3f}, |N| before scaling: {N_norm:.3f}, |N| after: {np.linalg.norm(N):.3f}")
        print(f"Step size ds: {self.ds:.4f} Å")
        
        # Show selected atom pair for Nl
        if selected_pair is not None:
            i, j = selected_pair
            print(f"\nSelected atom pair for Nl: atoms {i} and {j}, distance = {pair_distance:.3f} Å")
        
        # Show per-atom displacement magnitudes
        displacements = []
        for i in range(n_atoms):
            disp_vec = N[3*i:3*i+3]
            disp_mag = np.linalg.norm(disp_vec)
            actual_disp = self.ds * disp_mag
            is_mobile = "mobile" if (self.mobile_mask is None or self.mobile_mask[i]) else "fixed"
            displacements.append((i, disp_mag, actual_disp, is_mobile))
        
        # Print statistics
        mobile_disps = [d[2] for d in displacements if d[3] == "mobile"]
        if mobile_disps:
            print(f"Mobile atom displacements (Å): min={min(mobile_disps):.4f}, "
                  f"max={max(mobile_disps):.4f}, mean={np.mean(mobile_disps):.4f}")
        
        # Find top 2 atoms with largest displacement
        sorted_disps = sorted(displacements, key=lambda x: x[2], reverse=True)
        print(f"\nTop 2 atoms with largest displacement:")
        for idx, (atom_i, mag, actual, status) in enumerate(sorted_disps[:2]):
            vec = N[3*atom_i:3*atom_i+3]
            in_pair = " <- IN SELECTED PAIR" if (selected_pair and atom_i in selected_pair) else ""
            print(f"  #{idx+1}: Atom {atom_i:4d} | {status:6s} | |N_i|={mag:6.3f} | disp={actual:8.4f} Å{in_pair}")
            print(f"       Direction: [{vec[0]:7.3f}, {vec[1]:7.3f}, {vec[2]:7.3f}]")
        
        # Print detailed per-atom info (first 10 atoms)
        print("\nPer-atom displacement (first 10 atoms):")
        print("Atom | Status | |N_i| | ds*|N_i| (Å) | Direction")
        for i, mag, actual, status in displacements[:10]:
            vec = N[3*i:3*i+3]
            print(f"{i:4d} | {status:6s} | {mag:6.3f} | {actual:8.4f} | [{vec[0]:7.3f}, {vec[1]:7.3f}, {vec[2]:7.3f}]")
        if n_atoms > 10:
            print(f"... ({n_atoms - 10} more atoms)")
        print("="*60)
        
        return N
    
    def _generate_random_direction_python(self, atoms):
        """
        Generate random direction using user-defined Python function.
        Loads generate_random_direction.py from working directory.
        
        Returns:
            N: Random direction vector from user function
        """
        import sys
        import os
        
        # Get current working directory
        cwd = os.getcwd()
        script_path = os.path.join(cwd, 'generate_random_direction.py')
        
        if not os.path.exists(script_path):
            raise FileNotFoundError(
                f"Custom random direction script not found: {script_path}\n"
                f"When random_direction_mode='python', you must provide generate_random_direction.py "
                f"in the working directory with a function generate_random_direction(atoms) that returns N."
            )
        
        # Import user module
        import importlib.util
        spec = importlib.util.spec_from_file_location("user_random_direction", script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load {script_path}")
        
        user_module = importlib.util.module_from_spec(spec)
        sys.modules['user_random_direction'] = user_module
        spec.loader.exec_module(user_module)
        
        # Check if function exists
        if not hasattr(user_module, 'generate_random_direction'):
            raise AttributeError(
                f"Module {script_path} must define a function 'generate_random_direction(atoms)' that returns N."
            )
        
        # Call user function
        user_func = user_module.generate_random_direction
        N = user_func(atoms)
        
        # Validate output
        n_atoms = len(atoms)
        expected_size = 3 * n_atoms
        if not isinstance(N, np.ndarray):
            N = np.array(N)
        
        if N.shape != (expected_size,):
            raise ValueError(
                f"User function generate_random_direction must return a 1D numpy array of size {expected_size} (3*n_atoms), "
                f"but got shape {N.shape}"
            )
        
        print(f"\n=== Random Direction Generation (Python Mode) ===")
        print(f"Loaded custom function from: {script_path}")
        print(f"Direction vector |N| = {np.linalg.norm(N):.6f}")
        print("="*60)
        
        return N


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
                    # First step uses optimized direction from Step 2
                    N = optimized_direction
                else:
                    # Subsequent steps: refine direction using biased dimer rotation on current biased PES
                    # No new random direction - use current climbing direction as initial guess
                    N = self._biased_dimer_rotation(climb_atoms, N)
                
                # Apply mobility mask
                if self.mobile_mask is not None:
                    for i in range(len(climb_atoms)):
                        if not self.mobile_mask[i]:
                            N[3*i:3*i+3] = 0.0
                # Keep N magnitude from rotation and n_mobile scaling (do not renormalize here)
                
                # CRITICAL: Move structure along direction N before adding bias potential
                # This is Step 3 in the SSW paper: R^{n-1} displacement by ds along N_i^n
                # Add new Gaussian potential at the CURRENT position (before displacement)
                g_param = self._add_gaussian(climb_atoms, N)
                gaussian_params.append(g_param)
                # Debug: report Gaussian parameters
                d, R1, w = g_param
                print(f"Added Gaussian #{n}: w={w:.4f}, sigma≈{self.ds:.4f} Å, |d|={np.linalg.norm(d):.4f}")
                
                # THEN displace structure along direction N by ds
                climb_atoms_positions = climb_atoms.get_positions().flatten()
                climb_atoms_positions += self.ds * N
                climb_atoms.set_positions(climb_atoms_positions.reshape(-1, 3))
                
                # Calculate wall potential for current configuration
                wall_energy, wall_forces = calculate_wall_potential(climb_atoms, self.mobility_region, self.wall_strength, self.wall_offset)
                
                # Create biased calculator with all Gaussian potentials
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
                
                E_base, E_bias, E_wall = climb_atoms.calc.results['energy_components']
                print(f"Bias energy components: "
                    f"base = {E_base:.6f} eV, "
                    f"bias = {E_bias:.6f} eV, "
                    f"wall = {E_wall:.6f} eV"
                 )

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
        
        print(f"\n--- Dimer Rotation ---")
        print(f"Initial |N| = {norm_N:.6f}")
        print(f"Parameters: ΔR={delta_R:.5f} Å, θ_trial={np.degrees(theta_trial):.3f}°, max_iter={max_rotations}")
        
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
            
            print(f"  Iter {rotation_iter+1}: C={C:10.4f} eV/Å², |F_rot|={F_rot_mag:8.4f} eV/Å", end="")
            
            if F_rot_mag < f_rot_tol:
                print(" <- CONVERGED")
                break
            
            # Compute rotation angle using finite difference
            # Trial rotation: N' = N * cos(θ) + F_rot_normalized * sin(θ)
            if F_rot_mag > 1e-10:
                F_rot_normalized = F_rot / F_rot_mag
            else:
                print(" <- No rotation needed")
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
            
            print(f", θ_opt={np.degrees(theta_opt):7.3f}°")
            
            # Apply optimal rotation
            if abs(theta_opt) > 1e-6:
                N = N * np.cos(theta_opt) + F_rot_normalized * np.sin(theta_opt)
                N = N / np.linalg.norm(N)
            else:
                print("  -> Rotation angle too small, converged")
                break  # Converged
        
        print(f"Dimer rotation completed after {rotation_iter+1} iterations")
        return N



    def _add_gaussian(self, atoms, direction):
        """
        Generate new Gaussian potential parameters, according to the definition in the paper
        
        Parameters:
            atoms: Current atomic structure
            direction: Gaussian potential direction vector (already scaled by sqrt(n_mobile))
            
        Returns:
            tuple: (d, R1, w) - contains direction vector, reference position and height parameters
        """
        # Use direction as-is (already properly scaled)
        # DO NOT normalize - direction should preserve sqrt(n_mobile) scaling for proper projection
        d = direction.copy()
        
        # Use current atomic position as reference position R1
        R1 = atoms.positions.copy()
        
        # Adjust Gaussian potential height according to the climbing phase
        w = self.w
        
        return (d, R1, w)
    
    def get_minima_pool(self):
        """
        Get all found minima structures pool
        
        Returns:
            list: Contains all found minima Atoms structures list
        """
        return self.pool