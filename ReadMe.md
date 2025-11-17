# CoSMoS Global Optimization Algorithm Implementation

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

### GNU General Public License v3.0 Summary

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

lipai@mail.sim.ac.cn  
2025-11-14

## Project Overview
This project implements the CoSMoS (Core-Sampled Mobility Search) global optimization algorithm for potential energy surface exploration of atomic clusters, aggressive defects in crystals, surface structures, and interface structures. The algorithm combines biased potential energy techniques with mobility control mechanisms to efficiently find minimum energy points on potential energy surfaces.

## Installation Instructions
### Dependencies
- Python 3.8+
- numpy
- ase
- dscribe

### Installation Steps
1. Clone or download this project to your local machine
2. Run the installation script:
   ```bash
   sh install.sh
   ```

## Usage
### Basic Usage
```python
from cosmos_search import CoSMoSSearch
from ase import Atoms

# Create initial atomic structure
atoms = Atoms('H2O', positions=[[0,0,0], [0,1,0], [1,0,0]])

# Initialize calculator (example uses EMT calculator)
from ase.calculators.emt import EMT
calc = EMT()

# Create CoSMoS search instance
cosmos = CoSMoSSearch(
    initial_atoms=atoms,
    calculator=calc,
    H=14,               # Number of Gaussian potentials
    w=0.1,              # Gaussian potential height (eV)
    temperature=300,    # Temperature (K)
    mobility_control=True,  # Enable mobility control
    control_radius=5.0   # Core region radius (Å)
)

# Run search
cosmos.run(steps=100)
```

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

## Output
Optimization results will be saved in the `cosmos_output` directory, containing atomic structures and energy information for each step.

## References
1. Shang, R., & Liu, J. (2013). Stochastic surface walking method for global optimization of atomic clusters and biomolecules. The Journal of Chemical Physics, 139(24), 244104.
2. J. Chem. Theory Comput. 2012, 8, 2215

## Copyright
Copyright (c) 2025 CoSMoS Development Team. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
