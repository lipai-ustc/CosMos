# CoSMoS: Global Structure Search Program

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Input File Format](#input-file-format)
- [Examples](#examples)
- [Output](#output)
- [References](#references)
- [Copyright](#copyright)

## Overview
CoSMoS (Global Structure Search Program) is a tool for finding stable atomic structures using advanced optimization algorithms.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- ASE (Atomic Simulation Environment)

### Install from source
```bash
# Clone the repository
git clone https://github.com/lipai-ustc/cosmos.git
cd cosmos

# (Optional) Create and activate a clean Conda env
# conda create -n cosmos python=3.10 -y
# conda activate cosmos

# Install system-wide (recommended)
pip install .

# (Optional) Install in development mode (editable)
# pip install -e .

# (Optional) Install calculators (install as needed)
# pip install chgnet deepmd-kit
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
  - `type`: Potential type (eam/chgnet/deepmd)
  - `model`: Model file path or name (for all types)

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
| `calculator` | ASE calculator object | Required |
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
- `AlCu-EAM`: Aluminum-Copper alloy example with EAM potential
- `Au100-CHGNET`: Gold surface example with CHGNET potential
- `C60-deepmd`: Carbon-60 example with DeepMD potential

Each example contains:
- `input.json`: Calculation parameters
- `init.xyz`: Initial atomic structure
- Potential files (where required)

## Output
Optimization results will be saved in the `cosmos_output` directory (or custom directory specified in config), containing:
- `all_minima.xyz`: All found minimum energy structures in a single trajectory file
- `best_str.xyz`: The lowest energy structure found during the search
- `cosmos_log.txt`: Detailed log of the search process with energies at each step

## References
1. Shang, R., & Liu, J. (2013). Stochastic surface walking method for global optimization of atomic clusters and biomolecules. The Journal of Chemical Physics, 139(24), 244104.
2. J. Chem. Theory Comput. 2012, 8, 2215

## Copyright
Copyright (c) 2025 CoSMoS Development Team. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

