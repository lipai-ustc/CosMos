# CoSMoS: Global Structure Search Program

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Input File Format](#input-file-format)
- [Examples](#examples)
- [Output](#output)
- [References](#references)
- [License](#license)

## Overview
CoSMoS (Global Structure Search Program) is a tool for finding stable atomic structures using advanced optimization algorithms.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- ASE>=3.26.0 (Atomic Simulation Environment)
- dscribe>=2.1.0 (installed automatically via setup.py)
- **NequIP** (recommended for default calculator): `pip install nequip`
- Optional calculators: deepmd-kit, fairchem-core, chgnet (install as needed)

### Install from source
```bash
# Clone the repository
git clone https://github.com/lipai-ustc/CosMos.git
cd cosmos

# (Optional) Create and activate a clean Conda env
# conda create -n cosmos python=3.10 -y
# conda activate cosmos

# Install system-wide (recommended)
pip install .

# (Optional) Install in development mode (editable)
# pip install -e .

# Install NequIP (recommended for default calculator)
pip install nequip

# (Optional) Install other calculators as needed
# pip install deepmd-kit      # For DeepMD potential
# pip install fairchem-core   # For FAIRChem/OCP models
# pip install chgnet          # For CHGNet models
```

## Usage

### Basic Usage
CoSMoS requires an initial structure file and optionally a configuration file:
1. `init.xyz` - Initial structure file (required)
2. `input.json` - Configuration file (optional - uses defaults with NequIP from NEQUIP_MODEL)

Run with default NequIP calculator:
```bash
# Set NequIP model path
export NEQUIP_MODEL=/path/to/deployed_model.pth

# Run with minimal configuration
cosmos
```

Run with custom configuration:
```bash
# Basic execution with input.json
cosmos
```

### Example Execution
For the AlCu-EAM example:
```bash
# Navigate to example directory
cd examples/AlCu-EAM

# Generate initial structure (if needed)
python generate_structure.py

# Run structure search
cosmos
```

## Input File Format

### Low-Dimensional Structure Preparation
For low-dimensional systems (clusters / wires / slabs), **you must place the structural center near the center of the simulation box** before running CoSMoS. The internal geometry classification in `cosmos_search.py` detects vacuum layers by measuring distances from atoms to the cell boundaries. If a low-dimensional structure is located near the origin instead of the box center, the algorithm may misclassify the system (e.g., treating a cluster as bulk).

Practical recommendations:
- **0D cluster**: Place the cluster roughly at `(Lx/2, Ly/2, Lz/2)`.
- **1D wire**: The wire should extend along one axis, while its cross-section is centered in the box.
- **2D slab**: The slab should be centered along the vacuum direction.

If the structure is not centered, CoSMoS may choose inappropriate settings for mobile control and vacuum-axis handling.

### input.json Configuration

#### System and Potential
- `system`: System information
  - `name`: System name (optional)
  - `task`: Task type - `"global_search"` or `"structure_sampling"` (required)
  - `structure`: Path to initial structure file (optional, default: `"init.xyz"`)

- `potential`: Potential settings (optional - defaults to NequIP from NEQUIP_MODEL environment variable)
  - `type`: Potential type (nequip/eam/chgnet/deepmd/fairchem/vasp/lammps/python)
  - `model`: Model/configuration file path
    - For `nequip`: Path to deployed NequIP model file (e.g., `deployed_model.pth`)
    - For `eam`: Path to EAM potential file (e.g., `AlCu.eam.alloy`)
    - For `chgnet`: Model name (e.g., `pretrained`) or path
    - For `deepmd`: Path to DeepMD model file (e.g., `dp_model.pb`)
    - For `fairchem`: Pretrained model name (default: `EquiformerV2-31M-S2EF-OC20-All+MD`)
    - For `vasp`: Path to INCAR file (default: `INCAR`)
  - `device`: Device for computation (optional, default: `"cpu"`)
    - For GPU acceleration: `"cuda"`
  - `parameters`: Calculator-specific parameters (for lammps type)
  - For `type: "python"`: Create a `calculator.py` file in your working directory that defines a `calculator` variable with an ASE Calculator object

**Default Calculator:**
If no `potential` section is specified in `input.json`, CoSMoS will use NequIP as the default calculator:
```bash
# Set environment variable for default NequIP model
export NEQUIP_MODEL=/path/to/deployed_model.pth

# Run without potential configuration
cosmos
```

**Atomic Energy Calculator (for atomic-mode random directions):**
When using `random_direction.mode: "atomic"` or `"atomic_plus_nl"`, per-atom energies are required:
1. **Specified calculator**: If `atomic_energy_calculator` is configured in the `random_direction` section, uses that calculator
2. **Primary calculator fallback**: If not specified, uses the main potential calculator
3. **DeepMD**: Automatically uses custom wrapper with per-atom energy support
4. **Error handling**: Raises an error if the calculator does not support `get_potential_energies()`

Example atomic energy calculator configuration:
```json
"random_direction": {
  "mode": "atomic_plus_nl",
  "atomic_energy_calculator": {
    "type": "nequip",
    "model": "/path/to/atomic_energy_model.pth",
    "device": "cpu"
  }
}
```

**Supported Calculator Types:**

*NequIP (Equivariant Neural Network):*
```json
"potential": {
  "type": "nequip",
  "model": "deployed_model.pth",
  "device": "cpu"
}
```
- `model`: Path to deployed NequIP model file
- `device`: `"cpu"` or `"cuda"` (default: `"cpu"`)

NequIP is the **default calculator** when no potential is specified. Set the `NEQUIP_MODEL` environment variable:
```bash
export NEQUIP_MODEL=/path/to/deployed_model.pth
cosmos  # Will use NequIP automatically
```

*EAM Potential:*
```json
"potential": {
  "type": "eam",
  "model": "AlCu.eam.alloy"
}
```

*CHGNet (pretrained or custom):*
```json
"potential": {
  "type": "chgnet",
  "model": "pretrained"
}
```

*DeepMD:*
```json
"potential": {
  "type": "deepmd",
  "model": "dp_model.pb"
}
```

*FAIRChem (Open Catalyst Project):*
```json
"potential": {
  "type": "fairchem",
  "model": "/path/to/checkpoint.pt",
  "device": "cuda",
  "task_name": "oc20"
}
```
- `model`: Path to checkpoint file or pretrained model name (e.g., `EquiformerV2-31M-S2EF-OC20-All+MD`)
- `device`: `"cpu"` or `"cuda"` (default: `"cpu"`)
- `task_name`: Task name, usually `"oc20"` (default: `"oc20"`)

Example with custom checkpoint:
```json
"potential": {
  "type": "fairchem",
  "model": "/mnt/d/uma-s-1p1.pt",
  "device": "cuda"
}
```

*VASP (requires VASP installation):*
```json
"potential": {
  "type": "vasp",
  "model": "INCAR"
}
```
Reads VASP parameters from the specified INCAR file (default: `INCAR` in working directory). If file not found, uses default PBE parameters.

*LAMMPS:*
```json
"potential": {
  "type": "lammps",
  "commands": [...]
}
```

*Custom Python Calculator:*
```json
"potential": {
  "type": "python"
}
```
Requires a `calculator.py` file in the working directory.

#### Monte Carlo Layer
- `monte_carlo`: Monte Carlo parameters
  - `steps`: Total number of MC steps (required)
  - `temperature`: Temperature for Metropolis criterion in K (required)

#### Climbing Layer
- `climbing`: Climbing phase parameters
  - `gaussian`: Gaussian bias potential settings
    - `height`: Height of Gaussian bias potentials in eV (w parameter, default: 0.2)
    - `width`: Width of Gaussian potentials in Å (ds parameter, default: 0.2)
    - `Nmax`: Maximum number of Gaussians per climbing (H parameter, default: 20)
  - `random_direction`: Random direction generation parameters (optional)
    - `mode`: List of methods for generating random search directions (default: `["thermo","atomic"]`)
      - `"thermo"`: Use temperature-based random vectors (Boltzmann distribution)
      - `"atomic"`: Use `thermo` * energy-weighted random vectors based on per-atom energies
      - `"nl"`: Combine base random with local rigid movement (Nl)
      - `"element"`: Use element-weighted random vectors
      - `"python"`: Use custom user-defined function (see below)
    - `ratio`: List of weight configurations for combining multiple modes (default: `[[[0.5,0.5],1]]`)
    - `rotation_param`: Dimer rotation parameter (default: 10)
    - `element_weights`: Optional dict mapping element symbols to weight factors (e.g., `{"Cu": 1.5, "Al": 1.0}`)
    - `atomic_energy_calculator`: Calculator configuration for per-atom energies (only needed for `"atomic"` mode)
      - If not specified, uses the main potential calculator
      - Same format as `potential` section (with `type`, `model`, etc.)

#### Optimizer Layer
- `optimizer`: Local optimization settings
  - `max_steps`: Maximum optimization steps (default: 500)
  - `fmax`: Force convergence criterion in eV/Å (default: 0.05)

##### Custom Random Direction Generation
When `random_direction.mode` is set to `"python"`, you can provide your own random direction generation function:

1. Create a file named `generate_random_direction.py` in your working directory
2. Define a function with this signature:
```python
import numpy as np
from ase import Atoms

def generate_random_direction(atoms: Atoms) -> np.ndarray:
    """
    Generate custom random direction vector for SSW algorithm.
    
    Parameters:
        atoms: ASE Atoms object representing current structure
    
    Returns:
        N: 1D numpy array of size (3*n_atoms,) representing the direction vector
           Format: [x0, y0, z0, x1, y1, z1, ..., xn, yn, zn]
           Note: The vector will be automatically scaled by sqrt(N_mobile) to preserve
                 displacement magnitude in large systems
    """
    n_atoms = len(atoms)
    N = np.random.randn(3 * n_atoms)  # Your custom logic here
    return N
```

3. Set in `input.json`:
```json
"random_direction": {
  "mode": "python"
}
```

The function receives the current atomic structure and must return a flattened direction vector. You have full access to atomic positions, chemical symbols, energies, and any other ASE Atoms attributes to implement custom logic.

##### Complete Configuration Example
Here is a comprehensive example showing all major configuration options:

```json
{
  "system": {
    "name": "AlCu_alloy",
    "task": "global_search",
    "structure": "init.xyz"
  },
  "potential": {
    "type": "nequip",
    "model": "deployed_model.pth",
    "device": "cpu"
  },
  "monte_carlo": {
    "steps": 100,
    "temperature": 300
  },
  "climbing": {
    "gaussian_height": 0.2,
    "gaussian_width": 0.2,
    "max_gaussians": 20
  },
  "optimizer": {
    "max_steps": 500,
    "fmax": 0.05
  },
  "random_direction": {
    "mode": "atomic_plus_nl",
    "element_weights": {"Cu": 1.5, "Al": 1.0},
    "atomic_energy_calculator": {
      "type": "nequip",
      "model": "atomic_energy_model.pth",
      "device": "cpu"
    }
  },
  "mobile_control": {
    "mode": "region",
    "region_type": "sphere",
    "center": [5.0, 5.0, 5.0],
    "radius": 10.0,
    "wall_strength": 10.0,
    "wall_offset": 2.0
  },
  "output": {
    "directory": "cosmos_output"
  },
  "debug": false
}
```

#### Mobile Control Layer (Optional)
The mobile control feature allows you to constrain which atoms can move during optimization. **By default, all atoms are mobile.**

**Mode 1: All atoms mobile (default)**
```json
"mobile_control": {
  "mode": "all"
}
```
Or simply omit the `mobile_control` section entirely.

**Mode 2a: Control by mobile atom indices (indices_free)**
```json
"mobile_control": {
  "mode": "indices_free",
  "indices_free": [10, 20, 25],
  "wall_strength": 10.0,
  "wall_offset": 2.0
}
```
- `mode`: Set to `"indices_free"` to mark listed atoms as mobile (others fixed).
- `indices_free`: List of atom indices that are allowed to move (0-based).

**Mode 2b: Control by fixed atom indices (indices_fix)**
```json
"mobile_control": {
  "mode": "indices_fix",
  "indices_fix": [0, 1, 2],
  "wall_strength": 10.0,
  "wall_offset": 2.0
}
```
- `mode`: Set to `"indices_fix"` to mark listed atoms as fixed (others mobile).
- `indices_fix`: List of atom indices that are fixed (0-based).

**Mode 3: Control by spatial region**

*Sphere region:*
```json
"mobile_control": {
  "mode": "region",
  "region_type": "sphere",
  "center": [5.0, 5.0, 5.0],
  "radius": 10.0,
  "wall_strength": 10.0,
  "wall_offset": 2.0
}
```
- `mode`: Set to `"region"` for spatial control
- `region_type`: `"sphere"` for spherical region
- `center`: Center coordinates [x, y, z] in Å
- `radius`: Sphere radius in Å
- `wall_strength`: Repulsive wall strength to prevent atoms from leaving the region
- `wall_offset`: Mobile atoms can penetrate up to this distance beyond the boundary before feeling wall repulsion

*Slab region (between two parallel planes):*
```json
"mobile_control": {
  "mode": "region",
  "region_type": "slab",
  "origin": [0.0, 0.0, 0.0],
  "normal": [0, 0, 1],
  "min_dist": -5.0,
  "max_dist": 5.0,
  "wall_strength": 10.0,
  "wall_offset": 2.0
}
```
- `region_type`: `"slab"` for region between two planes
- `origin`: Reference point for plane distance calculation
- `normal`: Normal vector of the planes [x, y, z]
- `min_dist`: Minimum distance along normal direction (in Å)
- `max_dist`: Maximum distance along normal direction (in Å)

*Lower region (below threshold along an axis):*
```json
"mobile_control": {
  "mode": "region",
  "region_type": "lower",
  "axis": "z",
  "threshold": 10.0,
  "wall_strength": 10.0,
  "wall_offset": 2.0
}
```
- `region_type`: `"lower"` for atoms below/equal to threshold
- `axis`: Coordinate axis (`"x"`, `"y"`, or `"z"`)
- `threshold`: Threshold value in Å (atoms with axis coordinate ≤ threshold are mobile)

*Upper region (above threshold along an axis):*
```json
"mobile_control": {
  "mode": "region",
  "region_type": "upper",
  "axis": "z",
  "threshold": 10.0,
  "wall_strength": 10.0,
  "wall_offset": 2.0
}
```
- `region_type`: `"upper"` for atoms above/equal to threshold
- `axis`: Coordinate axis (`"x"`, `"y"`, or `"z"`)
- `threshold`: Threshold value in Å (atoms with axis coordinate ≥ threshold are mobile)

**Wall Potential:**
When using mobile control with `wall_strength > 0`, a quadratic repulsive wall potential prevents mobile atoms from penetrating too deep into immobile regions:
- Atoms can move up to `wall_offset` distance beyond the boundary without penalty
- Beyond that: `V_wall = 0.5 × wall_strength × overshoot²`
- This provides a soft boundary while maintaining flexibility

#### Output and Debugging
- `output`: Output settings
  - `directory`: Output directory name (default: `"cosmos_output"`)
  - `rd_xyz`: Enable output of random direction information (default: `false`)
  - `debug`: Enable debug mode for detailed logging (default: `false`)
    - When `true`: Logs step-by-step energy components (base, Gaussian bias, wall) during optimization to `cosmos_log.txt`
    - When `false`: Only logs final results and major events

**Debug Mode Example:**
```json
"output": {
  "directory": "cosmos_output",
  "debug": true
}

In debug mode, the log file will contain detailed information for each optimization step:
```
Step 1: E_total = -125.234567 eV, E_base = -125.500000 eV, E_bias = 0.250000 eV, E_wall = 0.015433 eV, fmax = 0.850000 eV/Å
Step 2: E_total = -125.456789 eV, E_base = -125.650000 eV, E_bias = 0.180000 eV, E_wall = 0.013211 eV, fmax = 0.420000 eV/Å
...
```

### Main Parameters
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `initial_atoms` | Initial atomic structure | Required |
| `calculator` | Calculator object | Required |
| `ds` | Step size (Å) | 0.2 |
| `H` | Number of Gaussian potentials | 20 |
| `w` | Gaussian potential height (eV) | 0.2 |
| `temperature` | Temperature (K) | Required |
| `mobile_control` | Mobile control configuration | `{"mode": "all"}` |

## Examples
Example directories are provided in the `examples/` folder:

### Standard Calculator Examples
- `AlCu-EAM`: Aluminum-Copper alloy with EAM potential
- `C60-deepmd`: Carbon-60 with DeepMD potential (requires dp_model.pb file)

### Custom Calculator Examples (using calculator.py)
- `C60-python-deepmd`: C60 example using DeepMD via custom calculator.py
- `C60-python-tersoff`: C60 example using Tersoff via custom calculator.py  
- `Au100-python-fairchem`: Gold surface example using FAIRChem via custom calculator.py

### Task Types
CoSMoS supports two task types:

1. **Global Search** (`global_search`):
   - Finds stable atomic structures across the potential energy surface
   - Suitable for discovering ground state and metastable structures

2. **Structure Sampling** (`structure_sampling`):
   - Samples atomic structures around existing minima
   - Suitable for generating configurations for statistical analysis or further simulations

**Note**: For custom calculator examples, the `calculator.py` file in each directory defines the ASE calculator. To use these examples:
1. Set `"type": "python"` in `input.json`
2. Ensure the calculator dependencies are installed (e.g., `pip install deepmd-kit` or `pip install fairchem-core`)
3. Modify `calculator.py` as needed for your system

Each example contains:
- `input.json`: Calculation parameters
- `generate_structure.py`: Script to generate initial structure
- `init.xyz`: Initial atomic structure (generated or provided)
- `calculator.py`: Custom calculator definition (for python type examples only)
- Model/potential files (where required, e.g., `dp_model.pb` for DeepMD)

## Output
Optimization results will be saved in the `cosmos_output` directory (or custom directory specified in config), containing:
- `all_minima.xyz`: All found minimum energy structures in a single trajectory file
- `best_str.xyz`: The lowest energy structure found during the search
- `cosmos_log.txt`: Detailed log of the search process with energies at each step

## References
1. Shang, R., & Liu, J. (2013). Stochastic surface walking method for global optimization of atomic clusters and biomolecules. The Journal of Chemical Physics, 139(24), 244104.
2. J. Chem. Theory Comput. 2012, 8, 2215

## License
This project is licensed under the GNU General Public License v3.0 (GPL-3.0).
You may copy, distribute, and modify this software under the terms of the GPL-3.0.
See https://www.gnu.org/licenses/gpl-3.0.html for the full license text.

