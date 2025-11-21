#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom calculator definition for C60

This file defines a calculator that will be loaded by CoSMoS when using type='python'.
The calculator variable must be defined and should be an ASE Calculator object.

This example demonstrates the calculator.py functionality.
"""

from ase.calculators.tersoff import Tersoff

# Define the calculator that will be used by CoSMoS
calculator = Tersoff.from_lammps("SiC.tersoff")
