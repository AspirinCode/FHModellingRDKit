from rdkit import Chem
import os, webbrowser
from utilities import *
import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator as MDC
import pickle


activeMolecules = Chem.SDMolSupplier("/home/knzk574/Desktop/Data/combinedactive.sdf")
inactiveMolecules = Chem.SDMolSupplier("/home/knzk574/Desktop/Data/combined_inactive.sdf")

# with open("smallInactiveSet", "r") as f:
#     smallInactiveSet = pickle.load(f)
# with open("smallActiveSet", "r") as f:
#     smallActiveSet= pickle.load(f)


def makeDescriptorTable(tableobj, outcome):
    descriptorTable = pd.DataFrame(tableobj)

    descriptorTable.columns = descriptorList

    descriptorTable['Outcome'] = outcome

    return descriptorTable


descriptorList = [desc[0] for desc in Descriptors.descList]

calc = MDC(descriptorList)

activesdesc = []

LOG_EVERY_N = 100

for i, mol in enumerate(activeMolecules):
        try:
            if mol.GetNumHeavyAtoms() > 0:
                activesdesc.append(MDC.CalcDescriptors(calc, mol))
                if (i % LOG_EVERY_N) == 0:
                    print str(i)+" molecules processed"
        except:
            print "Error while processing "

inactivedesc = []
for i, mol in enumerate(inactiveMolecules):
        try:
            if mol.GetNumHeavyAtoms() > 0:
                inactivedesc.append(MDC.CalcDescriptors(calc, mol))
                if (i%LOG_EVERY_N) == 0:
                    print str(i)+" molecules processed"
        except:
            print "Error while processing "
#inactivedesc = [MDC.CalcDescriptors(calc,i) for i in smallInactiveSet]


descDfActive = makeDescriptorTable(activesdesc, 1)
descDfInactive = makeDescriptorTable(inactivedesc, 0)

descDfAll = pd.concat([descDfActive, descDfInactive], ignore_index=True)

descDfAll.to_csv("descDFAll.csv")

print "Descriptors are ready. Moving on to prep the data"

#pickledump(descDfAll, "descdfAll")

