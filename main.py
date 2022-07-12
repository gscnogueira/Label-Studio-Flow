import json
import pickle

# import ktrain
# from ktrain import text as txt

import labelstudio as ls
from utils import gen_json

SERVER   = '164.41.76.30'
L_ID = '39'
U_ID = '40'
TOKEN    = 'bc36020e5d03487292cac63d82661daa12320042'

TDATA = 'train.conll'
VDATA = 'train.conll'


labeled_set   = ls.Project(SERVER, L_ID, TOKEN)
unlabeled_set = ls.Project(SERVER, U_ID, TOKEN)


print(labeled_set.list_snapshots())

# textos = unlabeled_set.get_txt()

# print(textos)

# data = [gen_json(texto, pred) for texto, pred in zip(textos, preds)]

# dump_set.import_data(data=data)






