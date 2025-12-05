import os
import json
import platform
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
import multiprocessing
import numpy as np
from ase import Atoms
from ase.io import read, write
from dscribe.descriptors import SOAP


def load_initial_structure(structure_path: str) -> Atoms:
    """Load initial structure from file"""
    return read(structure_path)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def get_version_info() -> Tuple[str, str, str, str, str]:
    """
    Get version information for cosmos and dependencies
    
    Returns:
        Tuple of (cosmos_version, python_version, ase_version, dscribe_version, header_string)
    """
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
    
    header = f"""cosmos {cosmos_version} ({os_name} {os_release})
executed on             {os_name} date {now_str}
running on    {total_cores} total cores
Python {python_ver}, ASE {ase_ver}, dscribe {dscribe_ver}"""
    
    return cosmos_version, python_ver, ase_ver, dscribe_ver, header


def print_version_header() -> str:
    """Print version header and return the header string"""
    _, _, _, _, header = get_version_info()
    print(header)
    return header


def load_potential(potential_config):
    """
    Automatically load different types of potential calculators based on configuration
    
    Parameters:
        potential_config: Dictionary containing potential configuration with keys:
            - 'type': Potential type ('eam', 'chgnet', 'deepmd', 'lammps', 'python')
            - 'model': Model file path or name (unified parameter for all types)
            - For 'python' type: loads calculator from calculator.py in working directory
    
    Returns:
        ASE Calculator object
    """
    pot_type = potential_config['type'].lower()
    
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
        from deepmd.calculator import DP
        model_path = potential_config.get('model')
        return DP(model=model_path)
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
    elif pot_type == 'vasp':
        from ase.calculators.vasp import Vasp
        # Parse VASP configuration from INCAR file
        incar_file = potential_config.get('model', 'INCAR')
        vasp_params = {}
        
        # Read INCAR file if it exists
        import os
        if os.path.exists(incar_file):
            with open(incar_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#') or line.startswith('!'):
                        continue
                    # Parse key = value pairs
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip().lower()
                        value = value.strip().split('#')[0].split('!')[0].strip()  # Remove inline comments
                        
                        # Convert value to appropriate type
                        try:
                            # Try integer
                            vasp_params[key] = int(value)
                        except ValueError:
                            try:
                                # Try float
                                vasp_params[key] = float(value)
                            except ValueError:
                                # Keep as string
                                vasp_params[key] = value
        else:
            # Default VASP parameters if INCAR not found
            vasp_params = {
                'xc': 'PBE',
                'encut': 400,
                'ediff': 1e-5,
                'kpts': (1, 1, 1),
                'ismear': 0,
                'sigma': 0.05
            }
            print(f"Warning: INCAR file '{incar_file}' not found. Using default VASP parameters.")
        
        return Vasp(**vasp_params)
    else:
        raise ValueError(f"Unsupported potential type: {pot_type}")


# Structure analysis and I/O utilities



def compute_soap_per_atom(atoms: Atoms, species: List[str], rcut: float = 6.0, nmax: int = 8, lmax: int = 6) -> np.ndarray:
    """Return per-atom SOAP descriptors (N_atoms x dim)."""
    soap = SOAP(species=species, periodic=True, r_cut=rcut, n_max=nmax, l_max=lmax)
    return soap.create(atoms)


def compute_sorted_structure_descriptor(atoms: Atoms, species: List[str], rcut: float = 6.0, nmax: int = 8, lmax: int = 6) -> np.ndarray:
    """Permutation-invariant structure descriptor by sorting per-atom SOAP rows by L2 norm and flattening."""
    per_atom = compute_soap_per_atom(atoms, species, rcut=rcut, nmax=nmax, lmax=lmax)
    norms = np.linalg.norm(per_atom, axis=1)
    order = np.argsort(norms)
    sorted_rows = per_atom[order]
    return sorted_rows.flatten()



def is_duplicate_by_desc_and_energy(new_atoms: Atoms,
                                    pool: List[Atoms],
                                    species: List[str],
                                    tol: float = 0.01,
                                    energy: Optional[float] = None,
                                    pool_energies: Optional[List[float]] = None,
                                    energy_tol: float = 1) -> bool:
    """
    Duplicate check combining permutation-invariant descriptor and energy gating.
    - Find closest structure in pool by descriptor distance.
    - If closest distance < tol, then require |ΔE| <= energy_tol (if energies provided).
    """
    if not pool:
        return False
    desc_new = compute_sorted_structure_descriptor(new_atoms, species)
    best_idx = -1
    best_dist = float('inf')
    for i, atoms in enumerate(pool):
        desc_old = compute_sorted_structure_descriptor(atoms, species)
        d = np.linalg.norm(desc_new - desc_old)
        if d < best_dist:
            best_dist = d
            best_idx = i
    if best_dist >= tol:
        return False
    if energy is not None and pool_energies is not None and 0 <= best_idx < len(pool_energies):
        return abs(energy - pool_energies[best_idx]) <= energy_tol
    return True

def write_structure_with_energy(filename: str, atoms: Atoms, energy: float) -> None:
    """
    Write structure and its energy information to XYZ file
    
    Parameters:
        filename: Output file path
        atoms: ASE Atoms object, structure to write
        energy: Structure's energy value (eV)
    """
    # Store energy information in atoms object's info dictionary
    atoms.info['energy'] = energy
    # Use ASE's write function to write XYZ file
    write(filename, atoms)
