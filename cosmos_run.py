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
    calculator_type = config.get('calculator_type', 'deepmd').lower()
    potential_path = config.get('potential_path')
    
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
    
    # Calculate box center coordinates
    cell = atoms.get_cell()
    box_center = np.mean(cell, axis=0) / 2
    print(f"Box center coordinates: {box_center} Å")
    
    # Load appropriate calculator based on configuration
    print(f"Loading {config.get('calculator_type', 'deepmd')} potential: {config.get('potential_path')}")
    calculator = load_calculator(config)
    
    # Initialize CoSMoS search with all parameters from configuration
    print("Initializing CoSMoS search...")
    ssw = CoSMoSSearch(
        initial_atoms=atoms,
        calculator=calculator,
        # Get all parameters from config with appropriate defaults
        ds=config.get('ds', 0.2),
        output_dir=config.get('output_dir', 'cosmos_results'),
        H=config.get('H', 14),
        temperature=config.get('temperature', 300),
        mobility_control=config.get('mobility_control', True),
        control_type=config.get('control_type', 'sphere'),
        control_center=box_center,
        control_radius=config.get('control_radius', 10.0),
        decay_type=config.get('decay_type', 'gaussian'),
        decay_length=config.get('decay_length', 5.0),
        # Add any other parameters from config
        **config.get('additional_parameters', {})
    )
    
    # Run CoSMoS global optimization
    steps = config.get('steps', 100)
    print(f"Starting CoSMoS global search with {steps} steps...")
    ssw.run(steps=steps)
    
    # Get results and output summary
    minima_pool = ssw.get_minima_pool()
    energies = [minima.get_potential_energy() for minima in minima_pool]
    
    print("\nCoSMoS search completed!")
    print(f"Found {len(minima_pool)} energy minimum structures")
    print(f"Lowest energy: {min(energies):.6f} eV")
    print(f"Results saved to: {config.get('output', 'cosmos_results')}")


if __name__ == '__main__':
    main()
