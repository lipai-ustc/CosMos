from fairchem.core import pretrained_mlip, FAIRChemCalculator
from ase.optimize import LBFGS

predictor = pretrained_mlip.load_predict_unit("/public/home/lipai/share/fairchem_models/uma-s-1p1.pt", device="cpu")
# replace "/mnt/d/uma-s-1p1.pt" with your checkpoint path

from ase import units
from ase.io import write
from ase.build import fcc100, add_adsorbate, molecule

calc = FAIRChemCalculator(predictor, task_name="oc20")

slab = fcc100("Cu", (3, 3, 3), vacuum=8, periodic=True)
adsorbate = molecule("CO")
add_adsorbate(slab, adsorbate, 2.0, "bridge")
slab.calc = calc

# Set up LBFGS dynamics object
opt = LBFGS(slab)
opt.run(0.05, 100)

write("CO@Cu.xyz",slab)
print(slab.get_potential_energies())
