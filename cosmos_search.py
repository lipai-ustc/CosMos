# cosmos_search.py (v2: faithful to Shang & Liu 2013)
# Reference 1: Shang, R., & Liu, J. (2013). Stochastic surface walking method for global optimization of atomic clusters and biomolecules. The Journal of Chemical Physics, 139(24), 244104.
# Reference 2: J. Chem. Theory Comput. 2012, 8, 2215
import os,sys
import numpy as np
from ase.io import write as ase_write
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
from bias_calculator import BiasCalculator
from cosmos_utils import is_duplicate_by_desc_and_energy, periodic_distance, print_xyz
class CoSMoSSearch:
    def __init__(
        self,
        task,                # Task type (e.g., 'global_search','structure_sampling')
        structure_info,      # Dict with 'atoms', 'geometry_type', 'vacuum_axes'
        calculator,          # ASE calculator object
        atomic_calculator,   # ASE calculator object for atomic energy calculation
        monte_carlo,         # Dict with 'steps', 'temperature'
        random_direction,    # Dict with 'mode', 'element_weights', 'atomic_calculator'
        gaussian,            # Dict with 'gaussian_height', 'gaussian_width', 'max_gaussians'
        optimizer,           # Dict with 'max_steps', 'fmax'
        mobile_control,    # Dict with 'mobile_atoms', 'mobile_region', 'wall_strength', 'wall_offset'
        output,          # Output directory
        **kwargs             # Additional parameters
    ):
        self.task = task     # Task type (e.g., 'global_search','structure_sampling')
        # Extract structure info
        self.atoms = structure_info['atoms'].copy()
        self.geometry_type = structure_info['geometry_type']
        self.vacuum_axes = structure_info['vacuum_axes']
        self.n_atoms = len(self.atoms)

        # Monte Carlo parameters
        self.temperature = monte_carlo['temperature']
        self.kB = 8.617333262e-5  # Boltzmann constant (eV/K)
        
        # Random direction parameters
        self.rd_mode = random_direction['mode']
        self.n_rd_mode = len(self.rd_mode)
        self.rd_ratio= random_direction['ratio']
        self.n_rd_scheme = len(self.rd_ratio)
        self.rd_ratio_scheme = [self.rd_ratio[i][-1] for i in range(self.n_rd_scheme)]
        self.rd_ratio_mode  =[self.rd_ratio[i][0] for i in range(self.n_rd_scheme)]
        self.element_weights = random_direction['element_weights']
        self.element_scales = np.repeat([self.element_weights.get(symbol, 1.0) for symbol in self.atoms.symbols], 3)  #[1,2,3] -> [1,1,1,2,2,2,3,3,3]
        self.quadra_a = random_direction['quadra_param']
        # Climbing parameters
        self.gaussian_width = gaussian['gaussian_width']   # Step size (also Gaussian width)
        self.gaussian_height = gaussian['gaussian_height']   # Gaussian potential height
        self.H = gaussian['max_gaussians']     # Max number of Gaussians
        
        # Optimizer parameters
        self.opt_fmax = optimizer['fmax']
        self.opt_max_steps = optimizer['max_steps']
        
        # Mobile control
        self.mobile_atoms = mobile_control['mobile_atoms']
        self.mobile_region = mobile_control['mobile_region']
        self.wall_strength = mobile_control['wall_strength']
        self.wall_offset = mobile_control['wall_offset']
        
        # Output and debug
        self.output_dir = output['directory']
        self.output_xyz = output['rd_xyz']
        self.debug = output['debug']
        
        # Handle additional parameters
        self.additional_params = kwargs
        
        # Initialize minimum pool
        os.makedirs(self.output_dir, exist_ok=True)
        self.pool = []  # Stores all found minimum energy structures
        self.real_energies = []  # Stores corresponding structure energies
        
        # Compute mobile mask once at initialization
        # mobile_atoms is now always provided as a list from cosmos_run (never None)
        # mobile_atoms is always provided by cosmos_run; if missing, this is a configuration error
        if self.mobile_atoms is None:
            raise ValueError("mobile_control['mobile_atoms'] must be provided and cannot be None.\n"
                             "Please check mobile_control configuration in input.json.")

        self.mobile_mask = np.zeros(self.n_atoms, dtype=bool)
        self.mobile_mask[self.mobile_atoms] = True
        # Cache number of mobile atoms
        self.n_mobile = len(self.mobile_atoms)
        # Apply FixAtoms constraint to immobile atoms
        # This ensures local minimization respects mobile constraints
        fixed_indices = [i for i in range(self.n_atoms) if not self.mobile_mask[i]]
        if len(fixed_indices) > 0:
            constraint = FixAtoms(indices=fixed_indices)
            self.atoms.set_constraint(constraint)
        
        # Set up calculators
        self.base_calc = calculator
        self.atomic_calc = atomic_calculator
        self.bias_calc = BiasCalculator(
            base_calculator=self.base_calc,
            mobile_mask=self.mobile_mask,
            ds=self.gaussian_width,
            mobile_region=self.mobile_region,
            wall_strength=self.wall_strength,
            wall_offset=self.wall_offset
        )

        # Initial structure optimization (real potential)
        if len(self.vacuum_axes) > 0:
            self.atoms.set_positions(self._position_to_center(self.atoms))
        self.atoms.calc = self.base_calc
        self._local_minimize(self.atoms)        
        self._add_to_pool(self.atoms)

    def _local_minimize(self, atoms):
        """
        Perform local structure optimization using LBFGS algorithm
        Complies with CoSMoS algorithm documentation steps 4 and 6, using limited-memory BFGS optimizer for efficiency
        
        Parameters:
            atoms: Atomic structure to optimize
            calc: Calculator for optimization, defaults to base_calc
            fmax: Convergence force threshold, defaults to class instance's fmax
        """
        if atoms.calc is None:
            raise ValueError("Atoms object must have a calculator assigned.")
        # Use LBFGS optimizer with custom logging if debug mode
        if self.debug and isinstance(atoms.calc, BiasCalculator):    # mute
            # Debug mode: log energy components at each step
           
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
                        E_total = self.atoms.get_potential_energy()
                        forces = self.atoms.get_forces()
                        fmax_current = (forces**2).sum(axis=1).max()**0.5
                        E_base, E_bias, E_wall = self.atoms.calc.results['energy_components']
                                                
                        print(f"  Step {self.step_count}: E_total = {E_total:.6f} eV, "
                                f"E_base = {E_base:.6f} eV, E_bias = {E_bias:.6f} eV, E_wall = {E_wall:.6f} eV, "
                                f"fmax = {fmax_current:.6f} eV/Å")
                    
                    return result
            print("Local minimization using LBFGS optimizer in debug mode")
            opt = DebugLBFGS(atoms, self, logfile=None)
        else:
            opt = LBFGS(atoms, logfile=None)

        if self.debug and isinstance(atoms.calc, BiasCalculator):
            atoms.get_potential_energy()
            print(f"Before opt  No.: {len(atoms.calc.gaussian_params)}  "
              f"E_base: {atoms.calc.results['E_base']:.3f} "
              f"F_base_max: {(atoms.calc.results['F_base']**2).sum(axis=1).max()**0.5:.3f}  "
              f"E_gauss: {atoms.calc.results['E_gaussian']:.3f}  "
              f"F_gauss_max: {(atoms.calc.results['F_gaussian']**2).sum(axis=1).max()**0.5:.3f}  "
              f"E_wall: {atoms.calc.results['E_wall']:.3f}  "
              f"F_wall_max: {(atoms.calc.results['F_wall']**2).sum(axis=1).max()**0.5:.3f}  "
              f"F_total_max: {(atoms.get_forces()**2).sum(axis=1).max()**0.5:.3f}  ")

        opt.run(fmax=self.opt_fmax,steps=self.opt_max_steps)
       
        # Output final energy components if available
        if self.debug and isinstance(atoms.calc, BiasCalculator):
            print(f"After opt   No.: {len(atoms.calc.gaussian_params)}  "
              f"E_base: {atoms.calc.results['E_base']:.3f} "
              f"F_base_max: {(atoms.calc.results['F_base']**2).sum(axis=1).max()**0.5:.3f}  "
              f"E_gauss: {atoms.calc.results['E_gaussian']:.3f}  "
              f"F_gauss_max: {(atoms.calc.results['F_gaussian']**2).sum(axis=1).max()**0.5:.3f}  "
              f"E_wall: {atoms.calc.results['E_wall']:.3f}  "
              f"F_wall_max: {(atoms.calc.results['F_wall']**2).sum(axis=1).max()**0.5:.3f}  "
              f"F_total_max: {(atoms.calc.results['forces']**2).sum(axis=1).max()**0.5:.3f}  "
              f"F_total_max: {(atoms.get_forces()**2).sum(axis=1).max()**0.5:.3f}  ")                      

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
        
        # Append to the combined trajectory file instead of individual files
        atoms_copy = atoms.copy()
        atoms_copy.info['energy'] = real_e
        atoms_copy.info['minima_index'] = len(self.pool) - 1
        
        # Append mode: add structure to existing file
        trajectory_file = os.path.join(self.output_dir, 'all_minima.xyz')
        ase_write(trajectory_file, atoms_copy, append=True)
        
        print(f"Found new minimum #{len(self.pool) - 1}: E = {real_e:.6f} eV")

    def _get_atomic_energies(self, atoms):
        """
        Returns:
            np.ndarray: Per-atom energies
        """
        # User explicitly specified calculator for atomic energy (already loaded)
        atoms_temp = atoms.copy()
        atoms_temp.calc = self.atomic_calc
        
        try:
            atomic_energies = atoms_temp.get_potential_energies()
            return atomic_energies
        except (AttributeError, NotImplementedError, RuntimeError) as e:
            raise RuntimeError(
                f"User-specified atomic energy calculator failed to compute per-atom energies: {e}\n"
                f"Please check 'random_direction.atomic_calc' configuration in input.json.")
    
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
        
        # Use pre-computed mobile_atoms from initialization
        mobile_indices = self.mobile_atoms
        
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
        normalized_energies = np.zeros(self.n_atoms)
        for i, symbol in enumerate(symbols):
            normalized_energies[i] = atomic_energies[i] - reference_energies[symbol]
        
        # Ensure all normalized energies are non-negative (they should be by construction)
        normalized_energies = np.maximum(normalized_energies, 0.0)
        
        # Calculate scales using exp(E_atom)
        # Add small offset to avoid issues with very small energies
        scales = 2 / (1 + np.exp(-4*normalized_energies))-1   # scales = 2*sigmoid(4*normalized_energies)
        # Set scales of masked (immobile) atoms to 0
        if self.mobile_mask is not None:
            scales[~self.mobile_mask] = 0.0
            # Normalize scales to have reasonable magnitude (mean = 1)
            # Only use mobile atoms for normalization
            mean_scale = np.mean(scales[mobile_indices])
            if mean_scale > 0:
                scales = scales / mean_scale

        #self._print_mobile(scales=scales,energies=atomic_energies,normalized_energies=normalized_energies)
        return scales

    def _generate_random_direction(self, atoms):
        """
        Generate random search direction, combining global soft movement and local rigid movement.
        Uses energy-based sampling: atoms with higher energy get larger random components.
        Complies with CoSMoS algorithm step 1: Generate initial random direction N⁰
        
        Returns:
            N: Normalized random direction vector
        """
        scheme=np.random.choice(np.arange(self.n_rd_scheme),p=self.rd_ratio_scheme)
        print(f"Random direction scheme {scheme} with modes {self.rd_mode} and weights {self.rd_ratio_mode[scheme]}")
        modes=self.rd_mode
        ratio_mode=self.rd_ratio_mode[scheme]

        N=np.zeros(3*self.n_atoms)
        for i,mode in enumerate(modes):
            if mode=='thermo':
                N_temp=self._generate_rd_thermo(atoms)
            elif mode=='atomic':
                N_temp=self._generate_rd_atomic(atoms)
            elif mode=='nl':
                N_temp=self._generate_rd_nl(atoms)
            elif mode=='element':
                N_temp=self._generate_rd_element(atoms)
            elif mode=='python':
                N_temp=self._generate_rd_python(atoms)
            else:
                raise ValueError(f"Unknown random direction mode: {mode}")
            N+=self._normalize(N_temp)*ratio_mode[i]
            if self.debug:
                print(f"Mode: {mode}, Ratio: {ratio_mode[i]:.3f}, |N_temp| : {np.linalg.norm(N_temp):.3f}")

        return self._remove_translation(atoms,N)
        
    def _generate_rd_thermo(self, atoms):
        """
        Generate random direction using thermodynamic method.
        """
        N = np.zeros(3 * self.n_atoms)        
        for i in range(self.n_atoms):
            if self.mobile_mask[i]:
                mass = atoms[i].mass 
                sigma = np.sqrt(self.kB * self.temperature / mass)
                N[3*i:3*i+3] = np.random.normal(0, sigma, 3)
        return N

    def _generate_rd_atomic(self, atoms):
        """
        Generate random direction using atomic energy method.
        """
        energy_scales = np.repeat(self._get_energy_based_scales(atoms), 3)  #[1,2,3] -> [1,1,1,2,2,2,3,3,3]
        N = energy_scales * self._generate_rd_thermo(atoms)
        return N

    def _generate_rd_element(self, atoms):
        """
        Generate random direction using element-wise energy method.
        """
        N=self.element_scales * self._generate_rd_thermo(atoms)
        return N
    
    def _generate_rd_nl(self, atoms):
        """
        Generate random direction using non-local bonding pattern method.
        """
        if self.n_mobile < 2:
            raise ValueError("Insufficient mobile atoms (need at least 2) for calculation.")
           
        max_attempts = 100
        attempts = 0
        N = np.zeros(3 * self.n_atoms)
        while attempts < max_attempts:
            # Select two non-neighboring atoms from mobile region
            i, j = np.random.choice(self.mobile_atoms, 2, replace=False)
            distance = atoms.get_distance(i, j, mic=True)
            if distance > 3.0:  # Only when atomic distance > 3Å
                # Generate local rigid movement direction according to equation (2) in paper
                # Nl = [qB - qA at position A, qA - qB at position B, 0, ...]
                qi = atoms.positions[i].flatten()
                qj = atoms.positions[j].flatten()  
                N[3*i:3*i+3] = qj - qi
                N[3*j:3*j+3] = qi - qj
                break
            attempts += 1
        
        if attempts == max_attempts:
            raise ValueError(f"Failed to find atom pair with distance > 3Å after {max_attempts} attempts, cannot generate local rigid movement direction.")

        return N
    
    def _generate_rd_python(self, atoms):
        """
        Generate random direction using user-defined Python function.
        Loads generate_random_direction.py from working directory.
        
        Returns:
            N: Random direction vector from user function
        """
        
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
        n_atoms = self.n_atoms
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
        atoms_temp = atoms.copy()
        atoms_temp.calc = self.base_calc
        return atoms_temp.get_potential_energy()
    
    def _get_bias_energy(self, atoms):
        """
        Get energy of structure on bias potential energy surface
        """
        atoms_temp = atoms.copy()
        atoms_temp.calc = self.bias_calc
        E=atoms_temp.get_potential_energy()
        if self.debug:
            E_base=atoms_temp.calc.results['E_base']
            E_bias=atoms_temp.calc.results['E_bias']
            print("bias flag: ",self.bias_calc.flag,"   quadratic:",self.bias_calc.quadra_params)
            print("E_base: ",E_base,"   E_bias: ",E_bias,"   E: ",E)
        return E

    def run(self, steps=100):
        """
        Run CoSMoS global search algorithm, strictly following steps in the paper
        """
        # Initial structure energy log (stdout is already tee'd to cosmos_log.txt by cosmos_run)
        #print(f"Initial structure: Energy = {self._get_real_energy(self.atoms):.6f} eV")
        
        # Initialize combined trajectory file (remove old one if exists)
        trajectory_file = os.path.join(self.output_dir, 'all_minima.xyz')
        if os.path.exists(trajectory_file):
            os.remove(trajectory_file)
        
        climb_info_file=open("climb.info","w")
        climb_info_file.write("#index before0/after1 d_climb_origin  angle  d_clime_base angle\n")
        # Initialize current structure as initial minimum structure
        init_atoms = self.atoms.copy()    # user provided initial structure (relaxed in self.__init__())
        basin_atoms= init_atoms.copy()     # current basin structure (relaxed)
        
        # Clean xyz directory if it exists
        if os.path.exists("xyz"):
            for file in os.listdir("xyz"):
                file_path = os.path.join("xyz", file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        elif self.output_xyz:
            os.makedirs("xyz")

        for step in range(steps):
            print(f"\n------------- CoSMoS Step {step + 1}/{steps} -------------")
            basin_energy = self._get_real_energy(basin_atoms)  # already relaxed
            N0 = self._generate_random_direction(basin_atoms)
            if self.output_xyz:
                print_xyz(basin_atoms,filename=f"climb_{step}.xyz",energy=basin_energy,bias_energy=0,N0=N0.reshape(-1,3))

            climb_atoms = basin_atoms.copy() # climbing structure
            gaussian_params = []
            for n in range(1, self.H + 1):
                # CRITICAL: Move structure along direction N before adding bias potential
                # This is Step 3 in the SSW paper: R^{n-1} displacement by ds along N_i^n
                # Add new Gaussian potential at the CURRENT position (before displacement)
                # THEN displace structure along direction N by ds
                # Locally optimize on modified potential energy surface
                N = self._bias_dimer_rotation_ase(climb_atoms, N0)
                gaussian_params.append((N.copy(),climb_atoms.positions.flatten(),self.gaussian_height))  #(d, R1, w)
                climb_atoms.calc = self.bias_calc
                climb_atoms.calc.reset_gaussians(gaussian_params)
                climb_atoms_positions = climb_atoms.get_positions().flatten() +self.gaussian_width * N
                climb_atoms.set_positions(climb_atoms_positions.reshape(-1, 3))
                self._local_minimize(climb_atoms)
                # Write to climbing.info file (file handle is kept open)
                if self.debug:
                    print(f"Added Gaussian #{n}: gaussian_height={self.gaussian_height:.4f}, sigma={self.gaussian_width:.4f} Å, |d|={np.linalg.norm(N):.4f}")
                    distance_0_org, angle_0_org = periodic_distance(init_atoms, climb_atoms, N)
                    distance_0_bas, angle_0_bas = periodic_distance(basin_atoms, climb_atoms, N)
                    climb_info_file.write(f"{step} 0 {distance_0_org:.4f} {angle_0_org:.4f} {distance_0_bas:.4f} {angle_0_bas:.4f}\n")
                    print(f"{step} 0 {distance_0_org:.4f} {angle_0_org:.4f} {distance_0_bas:.4f} {angle_0_bas:.4f}\n")

                tBE=climb_atoms.get_potential_energy()   # potential energy on bias potential energy surface
                climb_energy = self._get_real_energy(climb_atoms) # real energy on real potential energy surface

                if self.output_xyz:
                    print_xyz(climb_atoms,filename=f"climb_{step}.xyz",energy=climb_energy,bias_energy=tBE,N=N.reshape(-1,3))
                if self.debug:
                    # Write to climbing.info file (file handle is kept open)
                    distance_1_org, angle_1_org = periodic_distance(init_atoms, climb_atoms, N)
                    distance_1_bas, angle_1_bas = periodic_distance(basin_atoms, climb_atoms, N)
                    climb_info_file.write(f"{step} 1 {distance_1_org:.4f} {angle_1_org:.4f} {distance_1_bas:.4f} {angle_1_bas:.4f}\n")
                    print(f"{step} 1 {distance_1_org:.4f} {angle_1_org:.4f} {distance_1_bas:.4f} {angle_1_bas:.4f}\n")
                    climb_info_file.flush()
                
                # Algorithm Step 5: Check stopping condition
                # Stop if: (i) reached max Gaussians H, or (ii) structure relaxed back below starting energy
                if n >= self.H:
                    print(f"\n--- Climb end ---\n n_gaussian={n}, reached maximum Gaussians")
                    break
                if climb_energy <= basin_energy:
                    print(f"\n--- Climb end ---\n n_gaussian={n}, energy {climb_energy:.6f} eV <= basin {basin_energy:.6f} eV")
                    break
            
            # Algorithm Step 6: Remove all bias potentials and optimize on real potential energy surface
            new_basin_atoms=climb_atoms.copy()
            new_basin_atoms.calc = self.base_calc
            self._local_minimize(new_basin_atoms)
            new_basin_energy = self._get_real_energy(new_basin_atoms)

            if self.output_xyz:
                print_xyz(new_basin_atoms,filename=f"climb_{step}.xyz",energy=new_basin_energy,bias_energy=0,N0=N0.reshape(-1,3))
            
            # Algorithm Step 7: Use Metropolis criterion to accept or reject
            delta_E = new_basin_energy - basin_energy
            if delta_E > 0:
                accept_prob = np.exp(-delta_E / (self.kB * self.temperature))
            else:
                accept_prob = 1.0
            
            if np.random.rand() < accept_prob:
                print(f"Accept new basin structure: ΔE = {delta_E:.6f} eV, P = {accept_prob:.4f}")
                basin_atoms=new_basin_atoms
                if len(self.vacuum_axes) > 0:
                    basin_atoms.set_positions(self._position_to_center(basin_atoms))
                # Check if new structure is a duplicate
                if not is_duplicate_by_desc_and_energy(
                    new_atoms=basin_atoms,
                    pool=self.pool,
                    #species=self.soap_species if self.soap_species is not None else list(set(climb_atoms.get_chemical_symbols())),
                    energy=new_basin_energy,
                    pool_energies=self.real_energies,
                    energy_tol=0.5,
                    mobile_atoms=self.mobile_atoms,
                ):
                    # New unique structure found
                    self._add_to_pool(basin_atoms)   
                else:
                    # Duplicate structure, but still update init_atoms to explore from different point
                    # This prevents getting stuck in the same location
                    print("Structure is duplicate, not added to pool")
                    self._add_to_pool(basin_atoms) # still add to pool for the debug stage
                    # No need to update pool
            else:
                print(f"Reject new basin structure: ΔE = {delta_E:.6f} eV, P = {accept_prob:.4f}")
                # do not change basin_atoms, keep original basin_atoms as the current structure in Mento Carlo
                self._add_to_pool(new_basin_atoms)   
            
            # Output current step information
            print(f"Step {step+1}: Energy = {new_basin_energy:.6f} eV")

        print("\nCoSMoS search completed!")
        print(f"All {len(self.pool)} minima structures saved to: {os.path.join(self.output_dir, 'all_minima.xyz')}")
        
        # Save the lowest energy structure
        if self.pool and self.real_energies:
            min_idx = self.real_energies.index(min(self.real_energies))
            best_atoms = self.pool[min_idx].copy()
            best_energy = self.real_energies[min_idx]
            best_atoms.info['energy'] = best_energy
            best_atoms.info['minima_index'] = min_idx
            best_file = os.path.join(self.output_dir, 'best_str.xyz')
            ase_write(best_file, best_atoms)
            print(f"Lowest energy structure (E = {best_energy:.6f} eV) saved to: {best_file}")

        climb_info_file.close()
        return self.pool, self.real_energies

    def _position_to_center(self, atoms):
        if len(self.vacuum_axes) > 0:    # Pre-center structure along vacuum axes
            cell = atoms.get_cell()
            box_center = (cell[0] + cell[1] + cell[2]) / 2.0
            pos0 = atoms.get_positions()
            curr_center = pos0.mean(axis=0)
            shift = box_center - curr_center
            for i in range(3):
                if i not in self.vacuum_axes:
                    shift[i] = 0.0
            return pos0 + shift
        else:
            return atoms.get_positions()

    def _bias_dimer_rotation(self, atoms, N0):
        """
        Implement bias dimer rotation method according to SSW paper (Eq. 3-6)
        Uses proper dimer method to find the lowest curvature direction with bias potential
        
        Reference: Shang & Liu, J. Chem. Phys. 139, 244104 (2013)
        
        Parameters:
            atoms: Current atomic structure (at minimum R_m)
            initial_direction: Initial search direction N^0
        
        Returns:
            Optimized direction N^1 (normalized)
        """
        # Normalize initial direction
        norm_ini = np.linalg.norm(N0)
        N = N0.copy() / norm_ini
              
        # Dimer parameters from SSW paper
        delta_R = 0.02  # Dimer separation (typical: 0.02 Å)
        theta_trial = 0.5 * np.pi / 180.0  # Trial rotation angle (radians), ~1 degrees
        max_rotations = 100  # Maximum number of rotations
        f_rot_tol = 0.01  # Rotational force tolerance (eV/Å)

        # Current position (flattened)
        R0_flat = atoms.positions.flatten()
        
        print(f"\n--- Dimer Rotation ---")
        print(f"Parameters: ΔR={delta_R:.5f} Å, θ_trial={np.degrees(theta_trial):.3f}°, max_iter={max_rotations}")
        
        self.bias_calc.reset_quadra([self.quadra_a,R0_flat,N0])  #a=10
        # Iteratively rotate dimer to find optimal direction
        for rotation_iter in range(max_rotations):
            # Calculate dimer images: R_1 = R_0 + N * ΔR (Eq. 3)
            R1_flat = R0_flat + N * delta_R
            
            # Compute forces at R_1 (on real PES)
            atoms_temp1 = atoms.copy()
            atoms_temp1.set_positions(R1_flat.reshape(-1, 3))
            atoms_temp1.calc = self.bias_calc

            F1 = atoms_temp1.get_forces().flatten()
            print("quadra:",self.bias_calc.results['E_quadra'],self.bias_calc.results['F_quadra'].max(),self.bias_calc.results['F_quadra'].min())
            
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
            atoms_temp_trial = atoms.copy()
            atoms_temp_trial.set_positions(R1_trial_flat.reshape(-1, 3))
            atoms_temp_trial.calc = self.bias_calc
            F1_trial = atoms_temp_trial.get_forces().flatten()
            F0_trial_approx = -F1_trial
            C_trial = np.dot((F0_trial_approx - F1_trial), N_trial) / delta_R
            
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
            N = N * np.cos(theta_opt) + F_rot_normalized * np.sin(theta_opt)
            N = N / np.linalg.norm(N)
            
            # Check convergence after updating N
            if abs(theta_opt) < 1e-3:
                #print("  -> Rotation angle too small, converged")
                break  # Converged

        N = N * norm_ini
        if self.debug:
            print(f"Dimer rotation completed after {rotation_iter+1} iterations\n")
            print(f'N vector after rotation:\n|N|={np.linalg.norm(N):.6f}')
            for i in range(self.n_atoms):
                if i<10:
                    ns_mag = np.linalg.norm(N[3*i:3*i+3])
                    n_vec = N[3*i:3*i+3]
                    if(self.mobile_mask[i]):
                        print(f"Atom {i:3d}: N_i=[{n_vec[0]:8.4f}, {n_vec[1]:8.4f}, {n_vec[2]:8.4f}], |N_i|={ns_mag:.4f}")
        
        print()
        return self._remove_translation(atoms,N)   # Return optimized direction N^1 with original magnitude

    def _bias_dimer_rotation_ase(self,atoms,N0):
        """
        Utilize the built-in mask feature of dimer to optimize the eigenmode direction 
        to the lowest curvature direction considering only mobile atoms.
        
        Parameters:
        - atoms: ASE Atoms object
        - initial_direction: Initial direction vector (shape: 3*N,)
        - mobile_mask: Boolean array, True indicates mobile atoms
        - max_iterations: Maximum number of iterations
        - tol: Convergence tolerance
        Returns:
        - optimized_direction: Optimized eigenmode direction
        - curvature: Corresponding curvature value
        """
        from dimer.dimer import MinModeAtoms, DimerControl, DimerEigenmodeSearch

        atoms_temp=atoms.copy()
        atoms_temp.calc=self.bias_calc
        atoms_temp.calc.reset_quadra([self.quadra_a,atoms_temp.positions.flatten(),N0])  #a=10
        if self.debug:
            print("self.quadra_a=",self.quadra_a)

        dimer_mask = self.mobile_mask.tolist()

        initial_direction = N0.reshape(-1, 3) / np.linalg.norm(N0)
        masked_initial_direction = initial_direction * self.mobile_mask[:, None]
       
        # Set control parameters, including mask
        # Create DimerControl with logfile=None to disable output
        control = DimerControl(logfile=None, eigenmode_logfile=None)
        #control.set_parameter('dimer_separation', 0.02)
        control.set_parameter('mask', dimer_mask)  # Set mask parameter
        control.set_parameter('order', 1)  # We only need the first eigenmode
        
        # Create MinModeAtoms object
        min_mode_atoms = MinModeAtoms(atoms_temp, control=control, logfile=None)

        # Process initial direction, considering only mobile atoms
       
        # Set initial eigenmode
        min_mode_atoms.initialize_eigenmodes(eigenmodes=[masked_initial_direction])
        
        # Create eigenmode search object
        eigenmode_search = DimerEigenmodeSearch(min_mode_atoms)
        
        # Ensure eigenmode_search has no logfile
        eigenmode_search.logfile = None
        
        # Set control parameters for eigenmode search
        # Adjust these parameters to control convergence
        eigenmode_search.control.set_parameter('f_rot_min', 0.01)  # Convergence threshold
        eigenmode_search.control.set_parameter('f_rot_max', 0.1)  # Upper limit for stopping
        eigenmode_search.control.set_parameter('max_num_rot', 100)  # Maximum rotations
        
        # Use dimer's built-in method to converge to eigenmode

        if self.debug:
            atoms_temp2=atoms_temp.copy()
            atoms_temp2.set_positions(atoms_temp.positions+0.01*N0.reshape(-1,3))
            atoms_temp2.calc=self.bias_calc
            print("before rotation:  E_total=",atoms_temp2.get_potential_energy(), "E_base=",self.bias_calc.results['E_base']," E_quadra=",self.bias_calc.results['E_quadra'])

        eigenmode_search.converge_to_eigenmode()
        
        # Get the final converged eigenmode
        final_mode = eigenmode_search.eigenmode

        if self.debug:
            atoms_temp2=atoms_temp.copy()
            atoms_temp2.set_positions(atoms_temp.positions+0.01*final_mode)
            atoms_temp2.calc=self.bias_calc
            print("after rotation:  E_total=",atoms_temp2.get_potential_energy(), "E_base=",self.bias_calc.results['E_base']," E_quadra=",self.bias_calc.results['E_quadra'])
        
        # Update the eigenmode in MinModeAtoms
        min_mode_atoms.set_eigenmode(final_mode, order=1)
        
        # Get final curvature
        final_curvature = eigenmode_search.get_curvature()
        if self.debug:
            print(f"Final curvature: {final_curvature:.6f}")
        
        # Verify the final rotational force
        eigenmode_search.update_virtual_forces()
        final_rot_force = eigenmode_search.get_rotational_force()
        final_rot_force_norm = np.linalg.norm(final_rot_force)
        if self.debug:
            print(f"Final rotational force norm: {final_rot_force_norm:.6f}")
        
        # Return optimized direction and curvature
        optimized_direction = min_mode_atoms.get_eigenmode(order=1)
        #final_curvature = min_mode_atoms.get_curvature(order=1)
        
        #N=self._remove_translation_rotation(atoms,optimized_direction.flatten())
        N=optimized_direction.flatten()
        if(np.dot(N,N0)<0):
            N=-N
        return N

    def _remove_translation(self,atoms,N):
        """
        Remove translation from N vector
        """
        # do not remove translation and rotation for bulk structure
        if self.geometry_type == 'bulk' or len(self.mobile_atoms)!=self.n_atoms: 
            return self._normalize(N)
        # remove translation
        else:
            N_temp = N.reshape(-1,3)-N.reshape(-1,3).mean(axis=0)
            return self._normalize(N_temp.flatten())

    def _print_mobile(self, **kwargs):
        """
        Print values for mobile atoms from multiple lists with custom titles.
        **kwargs: Pairs of title=list, where each list contains values for all atoms.
        Example: self._print_mobile(energy=energies, force=forces)
        """
        for title, data_list in kwargs.items():
            if len(data_list) != self.n_atoms:
                raise ValueError(f"List '{title}' must have n_atoms elements")
        
        # Print values for mobile atoms
        for i in np.arange(self.n_atoms):
            if self.mobile_mask[i]:
                info = [f"AtomID: {i:3d}"]
                for title, data_list in kwargs.items():
                    info.append(f"{title} = {data_list[i]}")
                print(" | ".join(info))

    def _normalize(self,N):
        """
        Normalize N vector
        """
        N_mask=np.zeros_like(N)
        for i in range(self.n_atoms):
            if self.mobile_mask[i]:
                N_mask[3*i:3*i+3]=N[3*i:3*i+3]
        try:
            return N_mask / np.linalg.norm(N_mask)
        except:
            raise ValueError("Failed to normalize N vector")
