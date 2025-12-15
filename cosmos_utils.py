import os
import platform
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
import multiprocessing
import numpy as np
from ase import Atoms
from ase.io import read
from ase.calculators.calculator import Calculator, all_changes
from dscribe.descriptors import SOAP 

def get_version_info():
    # Get cosmos version
    try:
        from importlib.metadata import version as _pkg_version
        cosmos_version = _pkg_version('cosmos')
    except Exception:
        cosmos_version = 'unknown'
    
    # Get Python version
    python_ver = platform.python_version()
    
    # Get ASE version
    try:
        import ase
        ase_ver = getattr(ase, '__version__', 'unknown')
    except Exception:
        ase_ver = 'unknown'
    
    # Get dscribe version
    try:
        from importlib.metadata import version as _pkg_version
        dscribe_ver = _pkg_version('dscribe')
    except Exception:
        dscribe_ver = 'unknown'
    
    # Build header string
    os_name = platform.system()
    os_release = platform.release()
    now_str = datetime.now().strftime('%Y.%m.%d  %H:%M:%S')
    total_cores = multiprocessing.cpu_count()
    
    header=f"""cosmos {cosmos_version} ({os_name} {os_release})
executed on             {os_name} date {now_str}
running on    {total_cores} total cores
Python {python_ver}, ASE {ase_ver}, dscribe {dscribe_ver}"""
   
    return header
    
def load_potential(potential_config, custom_atomic=False):
    """
    Automatically load different types of potential calculators based on configuration
    
    Parameters:
        potential_config: Dictionary containing potential configuration with keys:
            - 'type': Potential type ('eam', 'chgnet', 'deepmd', 'lammps', 'python', 'nequip')
            - 'model': Model file path or name (unified parameter for all types)
            - For 'python' type: loads calculator from calculator.py in working directory
            - If empty/None: defaults to NequIP using NEQUIP_MODEL environment variable
    
    Returns:
        ASE Calculator object
    """
    
    try:
        pot_type = potential_config['type'].lower()
    except KeyError:
        raise ValueError("Potential configuration missing 'type' key")
    
    if pot_type == 'python':
        # Load custom calculator from calculator.py in current working directory
        import sys
        import os
        cwd = os.getcwd()
        calc_file = os.path.join(cwd, 'calculator.py')
        if not os.path.exists(calc_file):
            raise FileNotFoundError(
                f"calculator.py not found in {cwd}\n"
                f"When using type='python', you must provide a calculator.py file "
                f"that defines a 'calculator' variable with an ASE Calculator object."
            )
        # Temporarily add cwd to path
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        try:
            import calculator as calc_module
            # Force reload in case it was previously imported
            import importlib
            importlib.reload(calc_module)
            if not hasattr(calc_module, 'calculator'):
                raise AttributeError(
                    f"calculator.py must define a 'calculator' variable.\n"
                    f"Example: calculator = Tersoff()\n"
                )
            return calc_module.calculator
        except ImportError as e:
            raise ImportError(
                f"Failed to import calculator.py: {e}\n"
                f"Make sure calculator.py is valid Python code."
            )
        finally:
            # Clean up sys.path
            if cwd in sys.path:
                sys.path.remove(cwd)
    elif pot_type == 'eam':
        from ase.calculators.eam import EAM
        model_path = potential_config.get('model')
        return EAM(potential=model_path)
    elif pot_type == 'chgnet':
        from chgnet.model import CHGNet
        model_name = potential_config.get('model', 'pretrained')
        if model_name == 'pretrained':
            model = CHGNet.load()
        else:
            model = CHGNet.load(model_name)
        return model.get_calculator()
    elif pot_type == 'deepmd':
        if custom_atomic:    # custom 
            return DeepMDCalculatorWithAtomicEnergy(model=potential_config.get('model'))
        else:    
            from deepmd.calculator import DP
            return DP(model=potential_config.get('model'))
    elif pot_type == 'nep':
        from calorine.calculators import CPUNEP
        model_path = potential_config.get('model')
        return CPUNEP(model_path)
    elif pot_type == 'lammps':
        from ase.calculators.lammpslib import LAMMPSlib
        # Parse LAMMPS potential configuration
        lammps_commands = potential_config['commands']
        return LAMMPSlib(lmpcmds=lammps_commands)
    elif pot_type == 'fairchem':
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        # Parse FAIRChem configuration
        model_path = potential_config.get('model', 'EquiformerV2-31M-S2EF-OC20-All+MD')
        device = potential_config.get('device', 'cpu')
        task_name = potential_config.get('task_name', 'oc20')
        # Load pretrained model
        predictor = pretrained_mlip.load_predict_unit(model_path, device=device)
        # Create FAIRChem calculator
        return FAIRChemCalculator(predictor, task_name=task_name)
    elif pot_type == 'nequip':
        from nequip.ase import NequIPCalculator
        model_path = potential_config.get('model')
        device = potential_config.get('device', 'cpu')
        return NequIPCalculator.from_compiled_model(compile_path=model_path, device=device)
    elif pot_type == 'vasp':
        from ase.calculators.vasp import Vasp
        incar_file = potential_config.get('model', 'INCAR')
        # Try to read INCAR parameters if file exists
        vasp_params = {}
        import os
        if os.path.exists(incar_file):
            # Parse INCAR file to extract parameters
            with open(incar_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # Try to convert to appropriate type
                        try:
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except ValueError:
                            pass  # Keep as string
                        vasp_params[key.lower()] = value
        else:
            # Use default PBE parameters
            print(f"Warning: INCAR file '{incar_file}' not found. Using default PBE parameters.")
            vasp_params = {
                'xc': 'PBE',
                'prec': 'Accurate',
                'encut': 520,
                'ediff': 1e-5,
                'ismear': 0,
                'sigma': 0.05
            }
        
        return Vasp(**vasp_params)

# Structure analysis and I/O utilities

def compute_sorted_structure_descriptor(atoms: Atoms, mobile_atoms: Optional[List[int]] = None, rcut: float = 4.0, nmax: int = 5, lmax: int = 5) -> np.ndarray:
    """Permutation-invariant structure descriptor with element-wise grouping.

    - Mobile atoms (indices in `mobile_atoms`) are included; all others are ignored.
    - For each chemical element present in atoms, collect SOAP rows of that element,
      sort them by L2 norm, then flatten and concatenate across elements.
    """
    # Get unique species from atoms
    symbols = np.array(atoms.get_chemical_symbols())
    species = sorted(set(symbols))
    
    # Build SOAP descriptor for all atoms
    from dscribe.descriptors import SOAP
    soap = SOAP(species=species, periodic=True, r_cut=rcut, n_max=nmax, l_max=lmax)
    per_atom = soap.create(atoms)

    # Determine mobile mask from mobile_atoms
    n_atoms = len(atoms)
    if mobile_atoms is not None:
        mobile_mask = np.zeros(n_atoms, dtype=bool)
        mobile_mask[np.array(mobile_atoms, dtype=int)] = True
    else:
        mobile_mask = np.ones(n_atoms, dtype=bool)

    # Build descriptor grouped by element type
    blocks = []
    for elem in species:
        elem_mask = (symbols == elem) & mobile_mask
        if not np.any(elem_mask):
            continue
        elem_desc = per_atom[elem_mask]
        norms = np.linalg.norm(elem_desc, axis=1)
        order = np.argsort(norms)
        sorted_rows = elem_desc[order]
        blocks.append(sorted_rows.flatten())

    if not blocks:
        # No mobile atoms; return empty descriptor
        return np.array([])

    return np.concatenate(blocks)

def is_duplicate_by_desc_and_energy(new_atoms: Atoms,
                                    pool: List[Atoms],
                                    #species: List[str],
                                    tol: float = 0.1,
                                    energy: Optional[float] = None,
                                    pool_energies: Optional[List[float]] = None,
                                    energy_tol: float = 1,
                                    mobile_atoms: Optional[List[int]] = None) -> bool:
    """
    Duplicate check combining permutation-invariant descriptor and energy gating.
    - Find closest structure in pool by descriptor distance.
    - If closest distance < tol, then require |ΔE| <= energy_tol (if energies provided).
    """
    if not pool:
        return False
    desc_new = compute_sorted_structure_descriptor(new_atoms, mobile_atoms=mobile_atoms)
    best_idx = -1
    best_dist = float('inf')
    for i, atoms in enumerate(pool):
        # Same mobile_atoms set applies to all structures in this run
        desc_old = compute_sorted_structure_descriptor(atoms, mobile_atoms=mobile_atoms)
        d = np.linalg.norm(desc_new - desc_old)
        if d < best_dist:
            best_dist = d
            best_idx = i
    if best_dist >= tol:
        return False
    if energy is not None and pool_energies is not None and 0 <= best_idx < len(pool_energies):
        return abs(energy - pool_energies[best_idx]) <= energy_tol
    return True

# --- Geometry and Mobility Utilities ---

def infer_geometry_type(atoms: Atoms, vacuum_threshold_angstrom: float = 3.0):
    """
    Infer geometry type based on absolute vacuum margins (Å) along each lattice axis.
    Returns a tuple: (geometry_type, vacuum_axes).
    vacuum_axes lists axes with significant vacuum on both sides.
    """
    pos = atoms.get_positions()  # Cartesian positions within the cell
    cell = atoms.get_cell()
    axis_lengths = np.array([np.linalg.norm(cell[0]), np.linalg.norm(cell[1]), np.linalg.norm(cell[2])])
    min_abs = pos.min(axis=0)
    max_abs = pos.max(axis=0)
    margin_low_abs = min_abs
    margin_high_abs = axis_lengths - max_abs
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


def get_mobility_atoms(atoms: Atoms, mobility_region) -> np.ndarray:
    """
    Compute boolean mask for mobile atoms.
    True = mobile, False = immobile
    """
    n_atoms = len(atoms)
    if mobility_region is None:
        return np.ones(n_atoms, dtype=bool)

    positions = atoms.get_positions()
    mask = np.zeros(n_atoms, dtype=bool)
    if mobility_region['type'] == 'sphere':
        center = np.array(mobility_region['center'])
        radius = mobility_region['radius']
        distances = np.linalg.norm(positions - center, axis=1)
        mask = distances <= radius
    elif mobility_region['type'] == 'slab':
        normal = np.array(mobility_region['normal'])
        normal = normal / (np.linalg.norm(normal) or 1.0)
        origin = np.array(mobility_region['origin'])
        min_dist = mobility_region['min_dist']
        max_dist = mobility_region['max_dist']
        vectors = positions - origin
        distances = np.dot(vectors, normal)
        mask = (distances >= min_dist) & (distances <= max_dist)
    elif mobility_region['type'] in ('lower', 'upper'):
        axis = mobility_region['axis'].lower()
        threshold = mobility_region['threshold']
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        if axis not in axis_map:
            raise ValueError(f"Invalid axis '{axis}'. Must be 'x', 'y', or 'z'.")
        axis_index = axis_map[axis]
        coords = positions[:, axis_index]
        if mobility_region['type'] == 'lower':
            mask = coords <= threshold
        elif mobility_region['type'] == 'upper':
            mask = coords >= threshold
    return np.where(mask)[0].tolist()


def calculate_wall_potential(atoms: Atoms, mobility_region, wall_strength: float, wall_offset: float):
    """
    Calculate wall potential energy and forces for mobile atoms relative to mobility_region.
    Returns (wall_energy, wall_forces_flat)
    """
    if wall_strength == 0 or mobility_region is None:
        return 0.0, np.zeros(3 * len(atoms))
    positions = atoms.get_positions()
    n_atoms = len(atoms)
    wall_energy = 0.0
    wall_forces = np.zeros(3 * n_atoms)
    mobile_mask = get_mobility_mask(atoms, None, mobility_region) if mobility_region else np.ones(n_atoms, dtype=bool)
    if mobility_region['type'] == 'sphere':
        center = np.array(mobility_region.get('center', [0, 0, 0]))
        radius = mobility_region.get('radius', 0.0)
        for i in range(n_atoms):
            if not mobile_mask[i]:
                continue
            delta_r = positions[i] - center
            dist = np.linalg.norm(delta_r)
            if dist > radius + wall_offset:
                overshoot = dist - radius - wall_offset
                wall_energy += 0.5 * wall_strength * overshoot ** 2
                direction = delta_r / dist if dist > 0 else np.zeros(3)
                force = -wall_strength * overshoot * direction
                wall_forces[3*i:3*i+3] = force
    elif mobility_region['type'] == 'slab':
        normal = np.array(mobility_region.get('normal', [0, 0, 1]))
        normal = normal / (np.linalg.norm(normal) or 1.0)
        origin = np.array(mobility_region.get('origin', [0, 0, 0]))
        min_dist = mobility_region.get('min_dist', -5.0)
        max_dist = mobility_region.get('max_dist', 5.0)
        for i in range(n_atoms):
            if not mobile_mask[i]:
                continue
            delta_r = positions[i] - origin
            proj_dist = np.dot(delta_r, normal)
            if proj_dist < min_dist - wall_offset:
                overshoot = (min_dist - wall_offset) - proj_dist
                wall_energy += 0.5 * wall_strength * overshoot ** 2
                force = wall_strength * overshoot * normal
                wall_forces[3*i:3*i+3] = force
            elif proj_dist > max_dist + wall_offset:
                overshoot = proj_dist - (max_dist + wall_offset)
                wall_energy += 0.5 * wall_strength * overshoot ** 2
                force = -wall_strength * overshoot * normal
                wall_forces[3*i:3*i+3] = force
    elif mobility_region['type'] in ('lower', 'upper'):
        axis = mobility_region.get('axis', 'z').lower()
        threshold = mobility_region.get('threshold', 0.0)
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis_index = axis_map.get(axis, 2)
        for i in range(n_atoms):
            if not mobile_mask[i]:
                continue
            coord = positions[i, axis_index]
            if mobility_region['type'] == 'lower':
                # For lower: atoms with coord <= threshold are mobile
                # Apply wall force if coord > threshold + wall_offset
                if coord > threshold + wall_offset:
                    overshoot = coord - (threshold + wall_offset)
                    wall_energy += 0.5 * wall_strength * overshoot ** 2
                    force_vec = np.zeros(3)
                    force_vec[axis_index] = -wall_strength * overshoot
                    wall_forces[3*i:3*i+3] = force_vec
            else:  # upper
                # For upper: atoms with coord >= threshold are mobile
                # Apply wall force if coord < threshold - wall_offset
                if coord < threshold - wall_offset:
                    overshoot = (threshold - wall_offset) - coord
                    wall_energy += 0.5 * wall_strength * overshoot ** 2
                    force_vec = np.zeros(3)
                    force_vec[axis_index] = wall_strength * overshoot
                    wall_forces[3*i:3*i+3] = force_vec
    return wall_energy, wall_forces


class DeepMDCalculatorWithAtomicEnergy(Calculator):
    """Wrapper for DeepMD calculator to enable per-atom energy calculation.
    
    DeepMD's official Python calculator does not expose per-atom energies by default.
    This wrapper extends the standard DP calculator to retrieve atomic energies.
    
    Reference: https://zhuanlan.zhihu.com/p/457374515
    
    Usage:
        calc = DeepMDCalculatorWithAtomicEnergy(model_path='model.pb')
        atoms.calc = calc
        atomic_energies = atoms.get_potential_energies()
    """
    
    name = "DP_AtomicEnergy"
    implemented_properties = ['energy', 'free_energy', 'forces', 'virial', 'stress', 'energies']
    
    def __init__(self, model: str, label: str = "DP_AtomicEnergy", type_dict: Optional[dict] = None, **kwargs) -> None:
        """Initialize DeepMD calculator with atomic energy support.
        
        Args:
            model: Path to the DeepMD model file (.pb)
            label: Calculator label
            type_dict: Mapping of element types and their numbers (optional)
        """
        from deepmd.infer import DeepPot
        from pathlib import Path
        
        Calculator.__init__(self, label=label, **kwargs)
        self.dp = DeepPot(str(Path(model).resolve()))
        
        if type_dict:
            self.type_dict = type_dict
        else:
            self.type_dict = dict(
                zip(self.dp.get_type_map(), range(self.dp.get_ntypes()))
            )
    
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """Calculate energy, forces, virial, stress, and per-atom energies."""
        from ase.calculators.calculator import PropertyNotImplementedError
        
        if atoms is not None:
            self.atoms = atoms.copy()
        
        # Prepare input
        coord = self.atoms.get_positions().reshape([1, -1])
        if sum(self.atoms.get_pbc()) > 0:
            cell = self.atoms.get_cell().reshape([1, -1])
        else:
            cell = None
        symbols = self.atoms.get_chemical_symbols()
        atype = [self.type_dict[k] for k in symbols]
        
        # Get fparam and aparam from atoms.info if available
        fparam = self.atoms.info.get('fparam', None)
        aparam = self.atoms.info.get('aparam', None)
        
        # Call DeepPot model to get all properties including atomic energies
        # DeepPot.eval returns: (energy, forces, virial, atomic_energy, atomic_virial)
        e, f, v, atomic_e, _ = self.dp.eval(
            coords=coord, 
            cells=cell, 
            atom_types=atype, 
            fparam=fparam, 
            aparam=aparam
        )
        
        # Store standard properties
        self.results['energy'] = e[0][0]
        self.results['free_energy'] = e[0][0]
        self.results['forces'] = f[0]
        self.results['virial'] = v[0].reshape(3, 3)
        
        # Store per-atom energies
        self.results['energies'] = atomic_e[0]
        
        # Convert virial into stress for lattice relaxation
        if cell is not None:
            # Stress = -virial / volume (tensile stress is positive)
            stress = -0.5 * (v[0].copy() + v[0].copy().T) / self.atoms.get_volume()
            # Voigt notation
            self.results['stress'] = stress.flat[[0, 4, 8, 5, 2, 1]]
        elif 'stress' in properties:
            raise PropertyNotImplementedError
    
    def get_potential_energies(self, atoms=None):
        """Return per-atom energies."""
        if atoms is not None:
            self.calculate(atoms)
        return self.results.get('energies', np.zeros(len(self.atoms)))

