https://github.com/facebookresearch/fairchem

install
```
conda create -n fairchem-env python=3.12 -y
conda activate fairchem-env
pip install fairchem-core #  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Set the task for your application and calculate
oc20: use this for catalysis
omat: use this for inorganic materials
omol: use this for molecules
odac: use this for MOFs
omc: use this for molecular crystals

Relax an adsorbate on a catalytic surface  
```
from ase.build import fcc100, add_adsorbate, molecule
from ase.optimize import LBFGS
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
# or load the checkpoint file directly if you have it
# predictor = pretrained_mlip.load_predict_unit("your_path/uma-s-1p1.pt", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="oc20")

# Set up your system as an ASE atoms object
slab = fcc100("Cu", (3, 3, 3), vacuum=8, periodic=True)
adsorbate = molecule("CO")
add_adsorbate(slab, adsorbate, 2.0, "bridge")

slab.calc = calc

# Set up LBFGS dynamics object
opt = LBFGS(slab)
opt.run(0.05, 100)
```

Relax an inorganic crystal  
```
from ase.build import bulk
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
# or load the checkpoint file directly if you have it
# predictor = pretrained_mlip.load_predict_unit("your_path/uma-s-1p1.pt", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="omat")

atoms = bulk("Fe")
atoms.calc = calc

opt = FIRE(FrechetCellFilter(atoms))
opt.run(0.05, 100)
```

Run molecular MD
```
from ase import units
from ase.io import Trajectory
from ase.md.langevin import Langevin
from ase.build import molecule
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
# or load the checkpoint file directly if you have it
# predictor = pretrained_mlip.load_predict_unit("your_path/uma-s-1p1.pt", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="omol")

atoms = molecule("H2O")
atoms.calc = calc

dyn = Langevin(
    atoms,
    timestep=0.1 * units.fs,
    temperature_K=400,
    friction=0.001 / units.fs,
)
trajectory = Trajectory("my_md.traj", "w", atoms)
dyn.attach(trajectory.write, interval=1)
dyn.run(steps=1000)
```

Calculate a spin gap
```
from ase.build import molecule
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")

#  singlet CH2
singlet = molecule("CH2_s1A1d")
singlet.info.update({"spin": 1, "charge": 0})
singlet.calc = FAIRChemCalculator(predictor, task_name="omol")

#  triplet CH2
triplet = molecule("CH2_s3B1d")
triplet.info.update({"spin": 3, "charge": 0})
triplet.calc = FAIRChemCalculator(predictor, task_name="omol")

triplet.get_potential_energy() - singlet.get_potential_energy()
```

## To use off-line potential checkpoint
1. Download the potential checkpoint file according to the guidance on [FAIRChem releases page](https://github.com/facebookresearch/fairchem/releases).
2. Set the `potential_checkpoint` parameter in the `FAIRChemCalculator` to the path of the downloaded file.
