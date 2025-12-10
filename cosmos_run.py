#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CoSMoS Global Optimization Execution Script

Function: Reads structure file and potential model configuration from input.json, performs CoSMoS global search
Usage: cosmos [or] python cosmos_run.py
"""
import os
import sys
import json
import numpy as np
from ase.io import read
from cosmos_search import CoSMoSSearch
from cosmos_utils import load_potential, get_version_info, \
                         get_mobility_atoms, infer_geometry_type

class TeeLogger:
    """Redirect print output to both console and log file"""
    def __init__(self, log_file, mode='w'):
        self.terminal = sys.stdout
        self.log = open(log_file, mode)
    
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
    # Record start time
    import time
    start_time = time.time()
    
    # Get current working directory where input files should be
    cwd = os.getcwd()
    
    # 0. Load configuration file
    config_path = os.path.join(cwd, 'input.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")

    # 1. Get task type (required)
    sys_config = config.get('system')
    if not sys_config:
        raise ValueError("Configuration missing 'system' section")
    name = sys_config.get('name')
    task = sys_config.get('task').lower()
    if task not in ['global_search', 'structure_sampling']:
        raise ValueError("Invalid task type. Must be 'global_search' or 'structure_sampling'.")

    # 2. Read structure file (optional)
    structure_path = config.get('system', {}).get('structure', 'init.xyz')
    if not os.path.isabs(structure_path):
        structure_path = os.path.join(cwd, structure_path)
    try:
        atoms = read(structure_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Structure file not found: {structure_path}")
    except Exception as e:
        raise ValueError(f"Error reading structure file: {e}")
    
    geometry_type, vacuum_axes = infer_geometry_type(atoms) # Detect geometry type and prepare structure for search
    structure_info = {'atoms': atoms, 'geometry_type': geometry_type, 'vacuum_axes': vacuum_axes}
    
    # 3. Load potential calculator (required)
    potential_config = config.get('potential')
    potential_type = potential_config.get('type')
    if not potential_config or not potential_type:
        raise ValueError("Configuration missing 'potential' section")
    calculator = load_potential(potential_config,custom_atomic=False)

    # 4. Get Monte Carlo configuration (required)
    mc_config = config.get('monte_carlo')
    mc_steps = mc_config.get('steps')
    temperature = mc_config.get('temperature')
    if not mc_config or not mc_steps or not temperature:
        raise ValueError("Configuration missing 'monte_carlo' section")
    monte_carlo={'steps': mc_steps, 'temperature': temperature}
    
    # 5. Get random direction mode and parameters  (optional)
    rd_config = config.get('random_direction', {})
    rd_mode = rd_config.get('mode', 'base_plus_nl') # Default method according to the original SSW algorithm
    valid_modes = {'base',            # Temperature-based default scale (Boltzmann distribution)
                   'atomic',          # 'base' * atomic_energy_scale
                   'base_plus_nl',    # 'base' + Nl
                   'atomic_plus_nl',  # 'base' * atomic_energy_scale + Nl
                   'python'           # User-defined Python function ()
                   }
    if rd_mode not in valid_modes:
        raise ValueError(f"Invalid random direction mode: '{rd_mode}'. Must be one of {valid_modes}.")
    element_weights = rd_config.get('element_weights', {}) # additional weight based on element type
    atomic_energy_calculator_config = rd_config.get('atomic_energy_calculator', None)
    # Load atomic energy calculator if specified
    if atomic_energy_calculator_config:
        atomic_energy_calculator = load_potential(atomic_energy_calculator_config)
        # Verify it supports per-atom energy calculation
        if not hasattr(atomic_energy_calculator, 'get_potential_energies'):
            raise ValueError(
                f"Atomic energy calculator (type: {atomic_energy_calculator_config.get('type', 'unknown')}) does not support per-atom energies.\n"
                f"The calculator must have 'get_potential_energies' method.\n"
                f"Please specify a compatible calculator in 'random_direction.atomic_energy_calculator'."
            )
        print(f"Loaded user-specified atomic energy calculator: {atomic_energy_calculator_config.get('type', 'unknown')}")
    else:
        atomic_energy_calculator = load_potential(potential_config, custom_atomic=True)
        if not hasattr(atomic_energy_calculator, 'get_potential_energies') and 'atomic' in rd_mode:
            raise ValueError(
                f"Primary potential calculator does not support per-atom energies.\n"
                f"The calculator must have 'get_potential_energies' method.\n"
                f"Please specify a compatible calculator in 'random_direction.atomic_energy_calculator'."
            )
        print("No atomic_energy_calculator specified. Using primary potential calculator for per-atom energies.")
    random_direction={'mode': rd_mode, 'element_weights': element_weights, 'atomic_energy_calculator': atomic_energy_calculator}

    # 6. Get Climbing configuration (optional)
    climb_config = config.get('climbing',{})
    gaussian_height = climb_config.get('gaussian_height', 0.2)    # w parameter
    gaussian_width  = climb_config.get('gaussian_width', 0.2)     # ds parameter
    max_gaussians   = climb_config.get('max_gaussians', 20)       # H parameter
    climbing={'gaussian_height': gaussian_height, 'gaussian_width': gaussian_width, 'max_gaussians': max_gaussians}

    # 7. Get Optimizer configuration (optional)
    optimizer_config = config.get('optimizer',{})
    max_steps = optimizer_config.get('max_steps', 500)
    fmax = optimizer_config.get('fmax', 0.05)
    optimizer={'max_steps': max_steps, 'fmax': fmax}

    # 8. Get Mobility Control configuration and normalize to internal format (optional)
    raw_mc = config.get('mobility_control', {})
    mobility_mode   = raw_mc.get('mode', 'all')
    mobility_region = None
    wall_strength   = raw_mc.get('wall_strength', 10.0)
    wall_offset     = raw_mc.get('wall_offset', 2.0)

    if mobility_mode == 'all':
        mobile_atoms = np.arange(len(atoms), dtype=int).tolist()

    elif mobility_mode == 'indices_free':
        mobile_atoms = np.array(raw_mc.get('indices_free', []), dtype=int).tolist()

    elif mobility_mode == 'indices_fix':
        fixed = np.array(raw_mc.get('indices_fix', []), dtype=int)
        all_idx = np.arange(len(atoms), dtype=int)
        mobile_atoms = np.setdiff1d(all_idx, fixed, assume_unique=False).tolist()

    elif mobility_mode == 'region':
        region_type = raw_mc.get('region_type')
        if not region_type:
            raise ValueError("mobility_control mode 'region' requires 'region_type' to be specified")
        if region_type == 'sphere':
            center = raw_mc.get('center')
            radius = raw_mc.get('radius')
            if center is None or radius is None:
                raise ValueError("region_type 'sphere' requires 'center' and 'radius' to be specified")
            mobility_region = {
                'type': 'sphere',
                'center': np.array(center).tolist(),
                'radius': radius,
            }
        elif region_type == 'slab':
            origin = raw_mc.get('origin')
            normal = raw_mc.get('normal')
            min_dist = raw_mc.get('min_dist')
            max_dist = raw_mc.get('max_dist')
            if origin is None or normal is None or min_dist is None or max_dist is None:
                raise ValueError("region_type 'slab' requires 'origin', 'normal', 'min_dist', and 'max_dist' to be specified")
            mobility_region = {
                'type': 'slab',
                'origin': np.array(origin).tolist(),
                'normal': np.array(normal).tolist(),
                'min_dist': min_dist,
                'max_dist': max_dist,
            }
        elif region_type in ('lower', 'upper'):
            axis = raw_mc.get('axis')
            threshold = raw_mc.get('threshold')
            if axis is None or threshold is None:
                raise ValueError(f"region_type '{region_type}' requires 'axis' and 'threshold' to be specified")
            mobility_region = {
                'type': region_type,
                'axis': axis,
                'threshold': threshold,
            }
        else:
            raise ValueError(f"Unknown region_type: {region_type}.\n Valid options are 'sphere', 'slab', 'lower', and 'upper'.")
        
        mobile_atoms = get_mobility_atoms(atoms, mobility_region)  # Calculate mobile_atoms from mobility_region

    else:
        raise ValueError(f"Unknown mobility_control mode: {mobility_mode}")
        
    mobility_control = {      # Construct unified mobility_control_param
        'mobile_atoms': mobile_atoms,
        'mobility_region': mobility_region,
        'wall_strength': wall_strength,
        'wall_offset': wall_offset,    }
    
    # 9. Get output configuration with defaults (optional)
    output_config = config.get('output', {})
    output_dir = output_config.get('directory', 'cosmos_output')

    # 10. Get debug mode (optional)
    debug_mode = config.get('debug', False)    

    # Prepare log file and print all configuration parameters at the top
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'cosmos_log.txt')
    # Redirect stdout to TeeLogger (opens in write mode to clear existing content)
    sys.stdout = TeeLogger(log_path, mode='w')        
    
    print(get_version_info())    #header
    print('============================================================================================')   
    print('=                                CoSMoS Input Configuration                                =')
    print('============================================================================================') 
    print(f'  System name      : {name}')   
    print(f'  Task type        : {task}')
    print(f'  Structure file   : {structure_path}')
    print(f'  Geometry type    : {geometry_type}')
    print(f'  Potential type   : {potential_type}')
    print(f'  MC steps         : {mc_steps}')
    print(f'  Temperature (K)  : {temperature}')
    print(f'  Gaussian height w: {gaussian_height}')
    print(f'  Gaussian width ds: {gaussian_width}')
    print(f'  Max Gaussians H  : {max_gaussians}')
    print(f'  Optimizer steps  : {max_steps}')
    print(f'  Optimizer fmax   : {fmax}')
    print(f'  RD mode          : {rd_mode}')
    if element_weights:  # Empty dict evaluates to False
        print(f'  Element weights  : {element_weights}')
    if mobility_mode != 'all':
        print(f'  Mobility mode    : {mobility_mode}')
        print(f'  Mobile atoms     : {mobile_atoms}')       
        print(f'  Mobility region  : {mobility_region}')         
        print(f'  Wall strength    : {wall_strength}')
        print(f'  Wall offset      : {wall_offset}')
    else:
        print('  Mobility control : None')
    print(f'  Output dir       : {output_dir}')
    print(f'  Debug mode       : {debug_mode}')        

    print('\n\n=====================================    Start of CoSMoS Search    ==================================\n\n') 

    cosmos = CoSMoSSearch(
        task=task,
        structure_info=structure_info,
        calculator=calculator,
        monte_carlo=monte_carlo,
        random_direction=random_direction,
        climbing=climbing,
        optimizer=optimizer,
        mobility_control=mobility_control,
        output_dir=output_dir,
        debug=debug_mode,
    )
    
    # Run CoSMoS global optimization
    minima_pool, energies = cosmos.run(steps=mc_steps)
    
    print("\nCoSMoS search completed!")
    print(f"Found {len(minima_pool)} energy minimum structures")
    if energies:
        print(f"Lowest energy: {min(energies):.6f} eV")
    print(f"Results saved to: {output_dir}")
    print(f"All minima structures in: {os.path.join(output_dir, 'all_minima.xyz')}")
    print(f"Best structure saved to: {os.path.join(output_dir, 'best_str.xyz')}")

    print('\n\n=====================================    Finish of CoSMoS Search    ==================================\n\n')

    # Calculate and print total execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    print(f"\nTotal execution time: {hours:02d}:{minutes:02d}:{seconds:06.3f}")

if __name__ == '__main__':
    main()