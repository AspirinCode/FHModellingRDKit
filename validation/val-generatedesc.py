from rdkit import Chem
import os, webbrowser
from utilities import *
import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator as MDC
from multiprocessing import Pool


descriptorList = [desc[0] for desc in Descriptors.descList]

calc = MDC(descriptorList)

strlist = pickleload("lucstr.pkl")


def descCalc(mol):
    try:
        if mol.GetNumHeavyAtoms() > 0:
            return(MDC.CalcDescriptors(calc, mol))
    except:
        print "Error while processing "


def makeDescriptorTable(tableobj, outcome = 0):
    descriptorTable = pd.DataFrame(tableobj)

    descriptorTable.columns = descriptorList

    if outcome:
        descriptorTable['Outcome'] = outcome

    return descriptorTable


descList = []

LOG_EVERY_N = 100

for index,mol in enumerate(strlist):
    descList.append(descCalc(mol))
    if (index % LOG_EVERY_N) == 0:
        print str(index) + " molecules processed"

descDF = makeDescriptorTable(descList)

pickledump(descDF, "descForVal.pkl")