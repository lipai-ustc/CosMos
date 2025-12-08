#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CoSMoS Global Optimization Execution Script

Function: Reads structure file and potential model configuration from input.json, performs CoSMoS global search
Usage: cosmos [or] python cosmos_run.py
"""
import os
import sys
import numpy as np
from cosmos_search import CoSMoSSearch
from cosmos_utils import load_initial_structure, load_config, load_potential, get_version_info


class TeeLogger:
    """Redirect print output to both console and log file"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def main() -> None:
    # Get current working directory where input files should be
    cwd = os.getcwd()
    
    # Load configuration file
    config_path = os.path.join(cwd, 'input.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please create an input.json file in the current directory. See examples/ for templates."
        )
    
    # Load configuration
    config = load_config(config_path)
    
    # System banner (similar to VASP OUTCAR header)
    _, _, _, _, header = get_version_info()
    print(header)
    
    # Validate required configuration sections
    if 'potential' not in config:
        raise ValueError("Configuration missing 'potential' section")
    if 'monte_carlo' not in config:
        raise ValueError("Configuration missing 'monte_carlo' section")
    if 'climbing' not in config:
        raise ValueError("Configuration missing 'climbing' section")
    
    # Get structure file path from configuration
    structure_path = config.get('structure_path', os.path.join(cwd, 'init.xyz'))
    
    # Validate input files exist
    if not os.path.exists(structure_path):
        raise FileNotFoundError(
            f"Structure file not found: {structure_path}\n"
            f"Please provide an init.xyz file or specify 'structure_path' in input.json"
        )
    
    # Load structure
    atoms = load_initial_structure(structure_path)
    
    # Load calculator directly from load_potential
    # If no potential config provided, defaults to NequIP from NEQUIP_MODEL env var
    potential_config = config.get('potential', {})
    calculator = load_potential(potential_config)
    if calculator is None:
        raise ValueError("Failed to initialize calculator. Check your potential configuration.")
    
    # Get potential type for CoSMoS (handle default NequIP case)
    potential_type = potential_config.get('type', 'nequip').lower() if potential_config else 'nequip'
    
    # Get output configuration with defaults
    output_config = config.get('output', {})
    output_dir = output_config.get('directory', 'cosmos_output')
    debug_mode = output_config.get('debug', False)
    
    # Write header to log file and setup logging
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'cosmos_log.txt')
    with open(log_path, 'w') as f:
        f.write(header)
        f.write('\n\n')
    
    # Redirect stdout to both console and log file
    sys.stdout = TeeLogger(log_path)
    
    # Get Monte Carlo configuration
    mc_config = config['monte_carlo']
    mc_steps = mc_config.get('steps', 100)
    temperature = mc_config.get('temperature', 500)
    
    # Get Climbing configuration
    climb_config = config['climbing']
    gaussian_height = climb_config.get('gaussian_height', 0.1)  # w parameter
    gaussian_width = climb_config.get('gaussian_width', 0.2)    # ds parameter
    max_gaussians = climb_config.get('max_gaussians', 20)       # H parameter
    
    optimizer_config = climb_config.get('optimizer', {})
    max_steps = optimizer_config.get('max_steps', 500)
    fmax = optimizer_config.get('fmax', 0.05)
    
    # Random direction mode configuration
    rd_mode = climb_config.get('random_direction_mode', 'atomic_plus_nl')
    valid_modes = {'base', 'atomic', 'base_plus_nl', 'atomic_plus_nl', 'python'}
    valid_ints = {1: 'base', 2: 'atomic', 3: 'base_plus_nl', 4: 'atomic_plus_nl'}
    
    if isinstance(rd_mode, int):
        if rd_mode not in valid_ints:
            raise ValueError(
                f"Invalid random_direction_mode: {rd_mode}. "
                f"Must be one of {list(valid_ints.keys())} or {valid_modes}"
            )
        rd_mode = valid_ints[rd_mode]
    else:
        rd_mode = str(rd_mode).strip().lower()
        if rd_mode not in valid_modes:
            raise ValueError(
                f"Invalid random_direction_mode: '{rd_mode}'. "
                f"Must be one of {valid_modes} or {list(valid_ints.keys())}"
            )
    
    # Get Mobility Control configuration and normalize to internal format
    raw_mc = config.get('mobility_control', None)
    mobility_control_param = None
    if isinstance(raw_mc, dict):
        mode = raw_mc.get('mode', 'all')
        if mode == 'all':
            mobility_control_param = None
        elif mode == 'indices_free':
            idx = np.array(raw_mc.get('indices_free', []), dtype=int)
            mobility_control_param = {
                'mobile_atoms': idx.tolist(),
                'mobility_region': None,
                'wall_strength': raw_mc.get('wall_strength', 0.0),
                'wall_offset': raw_mc.get('wall_offset', 2.0),
            }
        elif mode == 'indices_fix':
            fixed = np.array(raw_mc.get('indices_fix', []), dtype=int)
            all_idx = np.arange(len(atoms), dtype=int)
            mobile = np.setdiff1d(all_idx, fixed, assume_unique=False)
            mobility_control_param = {
                'mobile_atoms': mobile.tolist(),
                'mobility_region': None,
                'wall_strength': raw_mc.get('wall_strength', 0.0),
                'wall_offset': raw_mc.get('wall_offset', 2.0),
            }
        elif mode == 'region':
            region_type = raw_mc.get('region_type', 'sphere')
            if region_type == 'sphere':
                mobility_control_param = {
                    'mobile_atoms': None,
                    'mobility_region': {
                        'type': 'sphere',
                        'center': np.array(raw_mc.get('center', [0, 0, 0])).tolist(),
                        'radius': raw_mc.get('radius', 10.0),
                    },
                    'wall_strength': raw_mc.get('wall_strength', 0.0),
                    'wall_offset': raw_mc.get('wall_offset', 2.0),
                }
            elif region_type == 'slab':
                mobility_control_param = {
                    'mobile_atoms': None,
                    'mobility_region': {
                        'type': 'slab',
                        'origin': np.array(raw_mc.get('origin', [0, 0, 0])).tolist(),
                        'normal': np.array(raw_mc.get('normal', [0, 0, 1])).tolist(),
                        'min_dist': raw_mc.get('min_dist', -5.0),
                        'max_dist': raw_mc.get('max_dist', 5.0),
                    },
                    'wall_strength': raw_mc.get('wall_strength', 0.0),
                    'wall_offset': raw_mc.get('wall_offset', 2.0),
                }
            elif region_type in ('lower', 'upper'):
                mobility_control_param = {
                    'mobile_atoms': None,
                    'mobility_region': {
                        'type': region_type,
                        'axis': raw_mc.get('axis', 'z'),
                        'threshold': raw_mc.get('threshold', 0.0),
                    },
                    'wall_strength': raw_mc.get('wall_strength', 0.0),
                    'wall_offset': raw_mc.get('wall_offset', 2.0),
                }
            else:
                raise ValueError(f"Unknown region_type: {region_type}")
        else:
            raise ValueError(f"Unknown mobility_control mode: {mode}")
    
    # Prepare NequIP config for atomic energy fallback (if available)
    nequip_fallback_config = None
    if 'nequip_fallback' in config:
        nequip_fallback_config = config['nequip_fallback']
    
    cosmos = CoSMoSSearch(
        initial_atoms=atoms,
        calculator=calculator,
        output_dir=output_dir,
        # Climbing parameters
        ds=gaussian_width,
        H=max_gaussians,
        w=gaussian_height,
        max_steps=max_steps,
        fmax=fmax,
        # Monte Carlo parameters
        temperature=temperature,
        # Mobility control parameters (new simplified format)
        random_direction_mode=rd_mode,
        mobility_control=mobility_control_param,
        # Potential type for calculator selection
        potential_type=potential_type,
        # NequIP config for atomic energy fallback
        nequip_config=nequip_fallback_config,
        # Debug mode
        debug=debug_mode
    )
    
    # Run CoSMoS global optimization
    print(f"Starting CoSMoS global search with {mc_steps} steps...")
    cosmos.run(steps=mc_steps)
    
    # Get results and output summary
    minima_pool = cosmos.get_minima_pool()
    energies = cosmos.real_energies  # Use pre-computed energies instead of recalculating
    
    print("\nCoSMoS search completed!")
    print(f"Found {len(minima_pool)} energy minimum structures")
    if energies:
        print(f"Lowest energy: {min(energies):.6f} eV")
    print(f"Results saved to: {output_dir}")
    print(f"All minima structures in: {os.path.join(output_dir, 'all_minima.xyz')}")
    print(f"Best structure saved to: {os.path.join(output_dir, 'best_str.xyz')}")


if __name__ == '__main__':
    main()