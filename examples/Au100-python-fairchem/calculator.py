from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.load_predict_unit("/mnt/d/uma-s-1p1.pt", device="cuda")
# replace "/mnt/d/uma-s-1p1.pt" with your checkpoint path

calculator = FAIRChemCalculator(predictor, task_name="oc20")

# Set the task for your application and calculate
# oc20: use this for catalysis
# omat: use this for inorganic materials
# omol: use this for molecules
# odac: use this for MOFs
# omc: use this for molecular crystals
