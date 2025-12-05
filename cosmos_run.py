#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CoSMoS Global Optimization Execution Script

Function: Reads structure file and potential model configuration from input.json, performs CoSMoS global search
Usage: cosmos [or] python cosmos_run.py
"""
import os
import numpy as np
from cosmos_search import CoSMoSSearch
from cosmos_utils import load_initial_structure, load_config, load_potential, get_version_info


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
    potential_config = config.get('potential', {})
    calculator = load_potential(potential_config)
    if calculator is None:
        raise ValueError("Failed to initialize calculator. Check your potential configuration.")
    
    # Get output configuration with defaults
    output_config = config.get('output', {})
    output_dir = output_config.get('directory', 'cosmos_output')
    
    # Write header to log file
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'cosmos_log.txt')
    with open(log_path, 'w') as f:
        f.write(header)
        f.write('\n\n')
    
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
    
    # Get Mobility Control configuration
    mobility_control_param = config.get('mobility_control', None)
    
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
        mobility_control=mobility_control_param
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