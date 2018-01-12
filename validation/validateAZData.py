from rdkit import Chem
import utilities

textfile = open("/home/knzk574/Desktop/LucStructures.txt")


molList = [Chem.MolFromSmiles(smile) for smile in textfile.readlines()]

utilities.pickledump(molList, "validation/lucstr.pkl")

