from rdkit import Chem
from rdkit.Chem import Descriptors
import pickle

activeMolecules = Chem.SDMolSupplier("/home/knzk574/Desktop/Data/combinedactive.sdf")
inactiveMolecules = Chem.SDMolSupplier("/home/knzk574/Desktop/Data/combined_inactive.sdf")

smallActiveSet = [activeMolecules[i] for i in range(1000)]
smallInactiveSet = [inactiveMolecules[i] for i in range(len(inactiveMolecules)/10)]

with open("smallInactiveSet", "w") as f:
    pickle.dump(smallInactiveSet, f)
with open("smallActiveSet", "w") as f:
    pickle.dump(smallActiveSet, f)





