#!/usr/bin/env python3
import sys
import os

# Add the CosMos directory to the Python path
sys.path.insert(0, '/public/home/lipai/share/apps/CosMos')

from cosmos_utils import load_potential

# Test loading the NEP potential
try:
    potential_config = {
        'type': 'nep',
        'model': 'examples/Si-I4-nep/Si.txt'
    }
    print("Testing NEP potential loading...")
    calculator = load_potential(potential_config)
    print("Success! NEP potential loaded correctly.")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()