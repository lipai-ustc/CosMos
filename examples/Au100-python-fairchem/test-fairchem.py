#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for FAIRChem calculator with ASE

Note: This is a demonstration. FAIRChem requires significant setup.
For CoSMoS integration, create a calculator.py file in your working directory
with the FAIRChem calculator configured as shown below, then set
"potential": {"type": "python"} in input.json
"""

from ase.build import fcc100, add_adsorbate, molecule
from ase.optimize import LBFGS

from fairchem.core import OCPCalculator

# Use OCPCalculator with a pretrained model checkpoint
# Note: First run will download the model
calc = OCPCalculator(
    model_name="EquiformerV2-31M-S2EF-OC20-All+MD",  # or another model
    cpu=True  # Set to False if using GPU
)

# Set up your system as an ASE atoms object
slab = fcc100("Cu", (3, 3, 3), vacuum=8, periodic=True)
adsorbate = molecule("CO")
add_adsorbate(slab, adsorbate, 2.0, "bridge")

slab.calc = calc

# Set up LBFGS dynamics object
opt = LBFGS(slab)
opt.run(0.05, 100)

print("✓ FAIRChem calculation completed successfully")

