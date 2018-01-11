from rdkit import Chem
from rdkit.Chem import Descriptors
import pickle

activeMolecules = Chem.SDMolSupplier("/home/knzk574/Desktop/Data/set3_actives.sdf")
inactiveMolecules = Chem.SDMolSupplier("/home/knzk574/Desktop/Data/set3_inactives.sdf")

smallActiveSet = [activeMolecules[i] for i in range(100)]
smallInactiveSet = [inactiveMolecules[i] for i in range(len(inactiveMolecules)/10000)]

with open("smallInactiveSet", "w") as wewe:
    pickle.dump(smallInactiveSet, wewe)
with open("smallActiveSet", "w") as dgrwe:
    pickle.dump(smallActiveSet, dgrwe)

print "Sets are ready, staring generation of descriptors"




