# cosmos_search.py (v2: faithful to Shang & Liu 2013)
# Reference 1: Shang, R., & Liu, J. (2013). Stochastic surface walking method for global optimization of atomic clusters and biomolecules. The Journal of Chemical Physics, 139(24), 244104.
# Reference 2: J. Chem. Theory Comput. 2012, 8, 2215
import os
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write
from dscribe.descriptors import SOAP

class BiasedCalculator(Calculator):
    """
    Biased potential energy calculator for modifying the Potential Energy Surface (PES) during CoSMoS climbing phase
    Adds multiple positive Gaussian bias potentials to the original energy surface to guide structural exploration
    Single Gaussian potential form: V_bias = w * exp(-(d · (R - R1))^2 / (2 * σ^2))
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, base_calculator, gaussian_params, ds=0.2, control_center=None, control_radius=10.0, mobility_weights=None, wall_strength=10.0, wall_offset=2.0):
        super().__init__()
        self.base_calc = base_calculator  # Original potential energy calculator
        self.gaussian_params = gaussian_params  # List of Gaussian parameters, each containing (d, R1, w)
        self.ds = ds  # Step size parameter, used for Gaussian potential width
        self.control_center = control_center  # Control center coordinates
        self.control_radius = control_radius  # Core region radius
        self.wall_offset = wall_offset  # Wall potential distance offset
        self.wall_strength = wall_strength  # Wall potential strength (eV/Å²)
        self.mobility_weights = mobility_weights

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

        # Calculate wall potential to prevent core region atoms from escaping
        wall_energy = 0.0
        wall_forces = np.zeros_like(R)
        wall_radius = self.control_radius + self.wall_offset
        n_atoms = len(atoms)

        if self.control_center is not None and self.mobility_weights is not None:
            for i in range(n_atoms):
                # Apply wall potential only to core region atoms
                if self.mobility_weights[i] == 1.0:
                    # Atomic position (unflattened)
                    pos = atoms.positions[i]
                    delta_R = pos - self.control_center
                    distance = np.linalg.norm(delta_R)

                    # Add quadratic repulsive potential when atom exceeds wall radius
                    if distance > wall_radius:
                        delta = distance - wall_radius
                        # Potential energy: V_wall = 0.5 * strength * delta²
                        wall_energy += 0.5 * self.wall_strength * (delta ** 2)

                        # Force: F = -strength * delta * (delta_R/distance), converted to flattened index
                        force = -self.wall_strength * delta * (delta_R / distance) if distance > 0 else 0
                        wall_forces[i*3 : (i+1)*3] = force

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
        self.results['energy'] = E0 + V_bias_total + wall_energy
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
        H=14,                # Number of Gaussian potentials
        w=0.1,               # Gaussian potential height (eV)
        temperature=300,     # Temperature (K) for Metropolis criterion
        # New parameters to align with cosmos_run.py
        radius=5.0,          # Core region radius
        decay=0.99,          # Decay coefficient
        wall_strength=0.0,   # Wall potential strength (eV/Å²)
        wall_offset=2.0,     # Wall potential distance offset (Å)
        # Mobility control parameters
        mobility_control=False,
        control_type='sphere',        # Control type: 'sphere' or 'plane'
        control_center=None,          # Control center coordinates
        control_radius=10.0,          # Control radius
        plane_normal=None,            # Plane normal vector
        decay_type='gaussian',        # Decay type
        decay_length=5.0,             # Decay length
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
        
        # Initialize mobility control parameters
        self.mobility_control = mobility_control
        self.control_radius = control_radius
        self.control_type = control_type
        self.decay_type = decay_type
        self.decay_length = decay_length
        
        if mobility_control:
            # Default to box center as control center
            self.control_center = control_center
            # Plane control parameters
            if control_type == 'plane' and plane_normal is None:
                self.plane_normal = np.array([0, 0, 1])  # Default z-axis normal
            else:
                self.plane_normal = plane_normal
            # Precompute initial mobility weights
            self._update_mobility_weights()
        else:
            # Set default values even when mobility control is disabled
            self.control_center = control_center
            self.plane_normal = plane_normal if plane_normal is not None else np.array([0, 0, 1])
            self.mobility_weights = np.ones(len(initial_atoms))
        
        # Handle additional parameters
        self.additional_params = kwargs
        
        os.makedirs(output_dir, exist_ok=True)
        self.pool = []  # Stores all found minimum energy structures
        self.real_energies = []  # Stores corresponding structure energies
        
        # Initial structure optimization (real potential)
        self.atoms.calc = self.base_calc
        self._local_minimize(self.atoms)
        self._add_to_pool(self.atoms)
        self._update_mobility_weights()

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

    def _update_mobility_weights(self):
        """
        Update mobility weights M based on atomic coordinates
        Calculate distance from atoms to core region based on control type:
        - 'sphere': Distance from atom to control center
        - 'plane': Distance from atom to plane (defined by control_center and plane_normal)
        """
        if not self.mobility_control:
            self.mobility_weights = np.ones(len(self.atoms))
            return
        
        positions = self.atoms.get_positions()
        distances = np.zeros(len(self.atoms))
        
        if self.control_type == 'sphere':
            # Calculate distance from atoms to sphere center
            distances = np.linalg.norm(positions - self.control_center, axis=1)
        elif self.control_type == 'plane':
            # Calculate distance from atoms to plane
            # Plane equation: normal · (x - point) = 0
            # Distance formula: |normal · (x - point)| / ||normal||
            vectors = positions - self.control_center
            normal = self.plane_normal / np.linalg.norm(self.plane_normal)
            distances = np.abs(np.dot(vectors, normal))
        
        # Calculate mobility weights (between 0-1)
        self.mobility_weights = np.zeros(len(self.atoms))
        for i, dist in enumerate(distances):
            if dist <= self.control_radius:
                # Inside core region, weight = 1
                self.mobility_weights[i] = 1.0
            else:
                # Outside core region, calculate weight based on decay type
                r = dist - self.control_radius
                if self.decay_type == 'linear':
                    # Linear decay
                    self.mobility_weights[i] = max(0, 1 - r/self.decay_length)
                elif self.decay_type == 'gaussian':
                    # Gaussian decay
                    self.mobility_weights[i] = np.exp(-(r**2)/(2*self.decay_length**2))
                else:
                    # Default no decay (abrupt cutoff)
                    self.mobility_weights[i] = 0.0

    def _generate_random_direction(self, atoms):
        """
        Generate random search direction, combining global soft movement and local rigid movement
        Complies with CoSMoS algorithm step 1: Generate initial random direction N⁰
        
        Returns:
            N: Normalized random direction vector
        """
        n_atoms = len(atoms)
        # Generate global soft movement direction Ns (follows Maxwell-Boltzmann distribution)
        # Use instance temperature instead of hardcoded value
        mass = 1.0  # Atomic mass unit
        scale = np.sqrt(self.k_boltzmann * self.temperature / mass)
        Ns = np.random.normal(0, scale, 3 * n_atoms)
        
        # Apply mobility weights: expand weights to 3N dimensions and apply to random direction
        if self.mobility_control:
            # Update weights (atomic positions may have changed)
            self._update_mobility_weights()
            # Apply each atom's weight to its 3 coordinate components
            atom_weights = np.repeat(self.mobility_weights, 3)
            Ns = Ns * atom_weights
        
        # Generate local rigid movement direction Nl (non-adjacent atom bonding pattern)
        Nl = np.zeros(3 * n_atoms)
        if n_atoms >= 2:
            # Randomly select two non-adjacent atoms
            # Filter atoms in core region
            # Use mobility_weights to identify core region atoms (weight = 1.0 indicates core region)
            core_atoms = np.where(self.mobility_weights == 1.0)[0].tolist()
            
            # Check if there are enough core region atoms
            if len(core_atoms) < 2:
                raise ValueError("Insufficient core region atoms (need at least 2) for calculation.")
            
            max_attempts = 50
            attempts = 0
            found = False
            
            while attempts < max_attempts and not found:
                # Select two non-neighboring atoms from core region
                indices = np.random.choice(core_atoms, 2, replace=False)
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
        # Apply mobility weights
        if self.mobility_control:
            atom_weights = np.repeat(self.mobility_weights, 3)
            direction = direction * atom_weights
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
        with open(os.path.join(self.output_dir, 'cosmos_log.txt'), 'w') as f:
            f.write("CoSMoS Search Log\n")
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
                
                # Apply mobility weights
                if self.mobility_control:
                    self._update_mobility_weights()
                    atom_weights = np.repeat(self.mobility_weights, 3)
                    N = N * atom_weights
                    N /= np.linalg.norm(N) if np.linalg.norm(N) > 0 else 1
                
                # CRITICAL: Move structure along direction N before adding bias potential
                # This is Step 3 in the SSW paper: R^{n-1} displacement by ds along N_i^n
                climb_atoms_positions = climb_atoms.get_positions().flatten()
                climb_atoms_positions += self.ds * N
                climb_atoms.set_positions(climb_atoms_positions.reshape(-1, 3))
                
                # Add new Gaussian potential at the displaced position
                g_param = self._add_gaussian(climb_atoms, N)
                gaussian_params.append(g_param)
                
                # Create biased calculator
                biased_calc = BiasedCalculator(
                    base_calculator=self.base_calc,
                    gaussian_params=gaussian_params,
                    ds=self.ds,
                    control_center=self.control_center,
                    control_radius=self.control_radius,
                    wall_strength=self.wall_strength,
                    wall_offset=self.wall_offset,
                    mobility_weights=self.mobility_weights
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
                if not is_duplicate_by_desc(climb_atoms, self.pool, self.soap_species, self.duplicate_tol):
                    # New unique structure found
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

# Auxiliary functions


def compute_soap_descriptor(atoms, species, rcut=6.0, nmax=8, lmax=6):
    """
    Calculate SOAP (Smooth Overlap of Atomic Positions) descriptor for atomic structure
    SOAP descriptor quantifies structural similarity and is sensitive to atomic arrangement and chemical environment
    
    Parameters:
    atoms: ASE Atoms object containing atomic structure information
    species: List containing possible element symbols in the system
    rcut: Cutoff radius controlling the range of atomic environment (default: 6.0 Å)
    nmax: Number of radial basis functions (default: 8)
    lmax: Maximum angular quantum number for spherical harmonics (default: 6)
    
    Returns:
    numpy array: Averaged SOAP descriptor vector
    """
    # Initialize SOAP descriptor calculator
    soap = SOAP(species=species, periodic=True, rcut=rcut, nmax=nmax, lmax=lmax)
    # Compute descriptor and average it over the atomic dimension to get the structure level descriptor
    return soap.create(atoms).mean(axis=0)  # Average descriptor


def is_duplicate_by_desc(new_atoms, pool, species, tol=0.01):
    """
    Check new structure for duplicate using SOAP descriptor
    Use descriptor vector Euclidean distance as similarity measure
    
    Parameters:
        new_atoms: ASE Atoms object, new structure to check
        pool: List containing known structures
        species: List containing possible element symbols in the system
        tol: Distance threshold, less than this value is considered a duplicate structure
    
    Returns:
        bool: True if duplicate structure, false otherwise
    """
    if not pool:
        return False
    # Compute new structure's SOAP descriptor
    desc_new = compute_soap_descriptor(new_atoms, species)
    # Compare with all structures in pool
    for atoms in pool:
        desc_old = compute_soap_descriptor(atoms, species)
        # Compute Euclidean distance of descriptor vectors
        if np.linalg.norm(desc_new - desc_old) < tol:
            return True
    return False


def write_minima(filename, atoms, energy):
    """
    Write minima structure and its energy information to XYZ file
    
    Parameters:
        filename: String, output file path
        atoms: ASE Atoms object, minima structure
        energy: Float number, structure's energy value(eV)
    """
    # Store energy information in atoms object's info dictionary
    atoms.info['energy'] = energy
    # Use ASE's write function to write XYZ file
    write(filename, atoms)