#!/bin/env python
import sys
from ase.io import read,write
from ase.optimize import LBFGS,MDMin,FIRE
from ase.constraints import ExpCellFilter
from calorine.calculators import CPUNEP

calc= CPUNEP("../Si.txt")

atoms=read("init.xyz")
atoms.calc=calc

opt=FIRE(atoms)
opt.run(fmax=0.01,steps=500)

f=open("OSZICAR",'w')    
f.write("E0 F= 0 0 %.4f"%(atoms.get_potential_energy()))
f.close()

write("CONTCAR",atoms)
write("CONTCAR.cif",atoms,format='cif')
write("contcar.xyz",atoms)

print(atoms.get_potential_energies())
