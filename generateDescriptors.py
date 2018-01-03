from rdkit import Chem
import os, webbrowser
from utilities import *
import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator as MDC
import pickle

with open("smallInactiveSet", "r") as f:
    smallInactiveSet = pickle.load(f)
with open("smallActiveSet", "r") as f:
    smallActiveSet= pickle.load(f)


def makeDescriptorTable(tableobj, outcome):
    descriptorTable = pd.DataFrame(tableobj)

    descriptorTable.columns = descriptorList

    descriptorTable['Outcome'] = outcome

    return descriptorTable


descriptorList = [desc[0] for desc in Descriptors.descList]

calc = MDC(descriptorList)

activesdesc = []
for i in smallActiveSet:
        try:
            if i.GetNumHeavyAtoms() > 0:
                activesdesc.append(MDC.CalcDescriptors(calc, i))
        except:
            print "Error while processing "

inactivedesc = []
for i in smallInactiveSet:
        try:
            if i.GetNumHeavyAtoms() > 0:
                inactivedesc.append(MDC.CalcDescriptors(calc, i))
        except:
            print "Error while processing "
#inactivedesc = [MDC.CalcDescriptors(calc,i) for i in smallInactiveSet]


descDfActive = makeDescriptorTable(activesdesc, 1)
descDfInactive = makeDescriptorTable(inactivedesc, 0)

descDfAll = pd.concat([descDfActive, descDfInactive], ignore_index=True)

pickledump(descDfAll, "descdfAll")



viewTable(descDfAll)