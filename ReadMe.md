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

# (Optional) Install calculators (install as needed)
# pip install deepmd-kit fairchem-core
```

## Usage

### Basic Usage
CoSMoS requires two input files in the working directory:
1. `input.json` - Calculation parameters configuration file
2. `init.xyz` - Initial structure file

Run the program with default files:
```bash
# Basic execution
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

### input.json Configuration

#### System and Potential
- `system`: System information
  - `name`: System name (optional)

- `potential`: Potential settings
  - `type`: Potential type (eam/chgnet/deepmd/python)
  - `model`: Model file path or name (for eam/chgnet/deepmd types)
  - For `type: "python"`: Create a `calculator.py` file in your working directory that defines a `calculator` variable with an ASE Calculator object

#### Monte Carlo Layer
- `monte_carlo`: Monte Carlo parameters
  - `steps`: Total number of MC steps (default: 100)
  - `temperature`: Temperature for Metropolis criterion in K (default: 300)

#### Climbing Layer
- `climbing`: Climbing phase parameters
  - `gaussian_height`: Height of Gaussian bias potentials in eV (w parameter, default: 0.1)
  - `gaussian_width`: Width of Gaussian potentials in Å (ds parameter, default: 0.2)
  - `max_gaussians`: Maximum number of Gaussians per climbing (H parameter, default: 14)
  - `optimizer`: Local optimization settings
    - `max_steps`: Maximum optimization steps (default: 500)
    - `fmax`: Force convergence criterion in eV/Å (default: 0.05)

#### Mobility Control Layer (Optional)
- `mobility_control`: Spatial constraint settings
  - `enabled`: Enable mobility control (default: False)
  - `control_type`: Type of control region ('sphere' or 'plane', default: 'sphere')
  - `control_center`: Center of control region [x, y, z] (default: box center)
  - `control_radius`: Radius of core region in Å (default: 10.0)
  - `decay_type`: Weight decay type ('gaussian', 'linear', or 'none', default: 'gaussian')
  - `decay_length`: Decay length scale in Å (default: 5.0)
  - `wall_potential`: Wall potential settings
    - `strength`: Wall potential strength in eV/Å² (default: 0.0)
    - `offset`: Wall offset distance in Å (default: 2.0)

#### Output
- `output`: Output settings
  - `directory`: Output directory name (default: 'cosmos_output')

### Main Parameters
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `initial_atoms` | Initial atomic structure | Required |
| `calculator` | Calculator object | Required |
| `ds` | Step size (Å) | 0.2 |
| `H` | Number of Gaussian potentials | 14 |
| `w` | Gaussian potential height (eV) | 0.1 |
| `temperature` | Temperature (K) | 300 |
| `mobility_control` | Whether to enable mobility control | False |
| `control_radius` | Core region radius (Å) | 10.0 |
| `wall_strength` | Wall potential strength (eV/Å²) | 10.0 |
| `wall_offset` | Wall potential distance offset (Å) | 2.0 |

## Examples
Example directories are provided in the `examples/` folder:

### Standard Calculator Examples
- `AlCu-EAM`: Aluminum-Copper alloy with EAM potential
- `C60-deepmd`: Carbon-60 with DeepMD potential (requires dp_model.pb file)

### Custom Calculator Examples (using calculator.py)
- `C60-python-deepmd`: C60 example using DeepMD via custom calculator.py
- `C60-python-tersoff`: C60 example using Tersoff via custom calculator.py  
- `Au100-python-fairchem`: Gold surface example using FAIRChem via custom calculator.py

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

