import re
import json
import pickle

import ktrain
from ktrain import text as txt

import labelstudio as ls

def gen_json(texto, pred):

    json_ = {"data":{} , "predictions":[]}
    json_["data"]["text"] = texto

    # print(json.dumps(json_))

    start = 0
    end   = 0
    label = None

    lista = []
    print(texto)
    print('-'*100)
    print(pred)

    while pred:

        if re.match(r'\s', texto[end]):
            end+=1
            continue

        token, tag  = pred.pop(0)

        if tag[0] != 'I' and end>0:
            if label:
                lista.append({'start':start,
                              'end' : end,
                              'text' : texto[start:end],
                              'label' : label})
            start = end

        end+=len(token)
        label = tag[2:]

    print('-'*100)
    print(lista)


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

# test_set.export_tasks(export_type='CSV', file='test.csv', id=3)

with open('preds.pickle', 'rb') as f:
    preds = pickle.load(f)

textos = test_set.get_txt()

gen_json(textos[1], preds[1])






