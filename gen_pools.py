import json
import pickle

from numpy.random import default_rng
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

rng = default_rng()
jsons = train_set.get_json()
texts = train_set.get_txt()

is_labeled = set(rng.choice(len(jsons), size=800, replace=False))

rotulados = []
n_rotulados = []

for i in range(len(jsons)):
    if i in is_labeled:
        rotulados.append(jsons[i])
    else:
        n_rotulados.append(texts[i])


n_rotulados = [sentence.replace('\n', '') for sentence in n_rotulados]
print(len(jsons))
print(len(texts))
print('-'*50)
print(len(rotulados))
print(len(n_rotulados))

json.dump(rotulados, open('rotulados.json', 'w'))

open('n_rotulados.txt', 'w').write("\n".join(n_rotulados))






