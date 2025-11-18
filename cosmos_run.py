#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CoSMoS Global Optimization Execution Script

Function: Reads structure file and potential model configuration from input.json, performs CoSMoS global search
Usage: python cosmos_run.py
"""
import os
import json
import numpy as np
from ase.io import read
from cosmos_search import CoSMoSSearch
from cosmos_utils import load_potential


def load_initial_structure(structure_path):
    """Load initial structure from file"""
    return read(structure_path)


def load_calculator(config):
    """Dynamically load calculator based on configuration"""
    # Get type and path from potential object
    potential_config = config.get('potential', {})
    calculator_type = potential_config.get('type', 'CHGNET').lower()
    potential_path = potential_config.get('filename')
    
    if not potential_path or not os.path.exists(potential_path):
        raise FileNotFoundError(f"Potential file not found: {potential_path}")
    
    # Import calculator class dynamically to avoid unnecessary dependencies
    if calculator_type == 'deepmd':
        from ase.calculators.deepmd import DeepMD
        return DeepMD(potential=potential_path)
    elif calculator_type == 'chgnet':
        from chgnet.calculator import CHGNetCalculator
        return CHGNetCalculator(model_path=potential_path)
    elif calculator_type == 'eam':
        from ase.calculators.eam import EAM
        return EAM(potential=potential_path)
    else:
        raise ValueError(f"Unsupported calculator type: {calculator_type}")


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    # Get current working directory where input files should be
    cwd = os.getcwd()
    
    # Only keep input.json configuration file parameter parsing
    config_path = os.path.join(cwd, 'input.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Get structure file path from configuration
    structure_path = config.get('structure_path', os.path.join(cwd, 'init.xyz'))
    
    # Validate input files exist
    if not os.path.exists(structure_path):
        raise FileNotFoundError(f"Structure file not found: {structure_path}")
    
    # Load structure
    atoms = load_initial_structure(structure_path)
    
    # Load calculator before initializing CoSMoSSearch
    calculator = load_calculator(config)
    if calculator is None:
        raise ValueError("Failed to initialize calculator")
    
    # Get output directory from config with default
    output_dir = config['output'].get('output_dir', 'cosmos_output')
    
    # Calculate default control center (box center)
    cell = atoms.get_cell()
    default_control_center = np.mean(cell, axis=0)/2 if np.any(cell) else np.array([5.0, 5.0, 5.0])
    
    cosmos = CoSMoSSearch(
        initial_atoms=atoms,
        calculator=calculator,
        output_dir=output_dir,  # Now properly defined
        # Basic search parameters (with defaults)
        H=config['cosmos_search'].get('H', 20),
        w=config['cosmos_search'].get('w', 0.1),
        ds=config['cosmos_search'].get('ds', 0.2),
        max_steps=config['cosmos_search'].get('max_steps', 1000),
        temperature=config['cosmos_search'].get('temperature', 300),
        radius=config['cosmos_search'].get('radius', 5.0),
        decay=config['cosmos_search'].get('decay', 0.99),
        # Basic wall potential parameters
        wall_strength=config['cosmos_search'].get('wall_strength', 0.0),
        wall_offset=config['cosmos_search'].get('wall_offset', 2.0),
        # Wall type control parameters (new)
        mobility_control=config['cosmos_search'].get('mobility_control', False),
        control_type=config['cosmos_search'].get('control_type', 'sphere'),  # Sphere/plane control
        control_center=config['cosmos_search'].get('control_center', default_control_center),
        control_radius=config['cosmos_search'].get('control_radius', 10.0),
        plane_normal=config['cosmos_search'].get('plane_normal', [0, 0, 1]),  # Plane normal vector
        decay_type=config['cosmos_search'].get('decay_type', 'gaussian'),
        decay_length=config['cosmos_search'].get('decay_length', 5.0),
        **config.get('additional_parameters', {})  # Add any other parameters from config
    )
    
    # Run CoSMoS global optimization
    steps = config.get('steps', 100)
    print(f"Starting CoSMoS global search with {steps} steps...")
    cosmos.run(steps=steps)
    
    # Get results and output summary
    minima_pool = cosmos.get_minima_pool()
    energies = [minima.get_potential_energy() for minima in minima_pool]
    
    print("\nCoSMoS search completed!")
    print(f"Found {len(minima_pool)} energy minimum structures")
    print(f"Lowest energy: {min(energies):.6f} eV")
    print(f"Results saved to: {config.get('output', 'cosmos_results')}")


if __name__ == '__main__':
    main()
