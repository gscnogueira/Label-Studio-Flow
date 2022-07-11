import json
import pickle

# import ktrain
# from ktrain import text as txt

import labelstudio as ls
from utils import gen_json

SERVER   = '164.41.76.30'
TRAIN_ID = '36'
TEST_ID  = '35'
DUMP_ID  = '37'
TOKEN    = 'bc36020e5d03487292cac63d82661daa12320042'

TDATA = 'train.conll'
VDATA = 'train.conll'


train_set = ls.Project(SERVER, TRAIN_ID, TOKEN)
test_set = ls.Project(SERVER, TEST_ID, TOKEN)
dump_set = ls.Project(SERVER, DUMP_ID, TOKEN)


textos = test_set.get_txt()

preds = pickle.load(open('preds.pickle', 'rb'))


data = [gen_json(texto, pred) for texto, pred in zip(textos, preds)]

dump_set.import_data(data=data)






