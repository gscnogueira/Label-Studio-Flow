import json
import pickle

import ktrain
from ktrain import text as txt

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


train_set.export_tasks(snapshot=1)

(trn, val, preproc) = txt.entities_from_conll2003(TDATA, val_filepath=VDATA)

model_bertimbau = txt.sequence_tagger(
    'bilstm-bert', preproc, verbose=0,
    bert_model='neuralmind/bert-base-portuguese-cased')

learner_bertimbau = ktrain.get_learner(
    model_bertimbau, train_data=trn,
    val_data=val, batch_size=128)

learner_bertimbau.fit(
    0.01, 1, cycle_len=5,
    checkpoint_folder='/tmp/saved_weights_bertimbau')

predictor_bertimbau = ktrain.get_predictor(
    learner_bertimbau.model, preproc)

textos = test_set.get_txt()

preds = predictor_bertimbau.predict(textos)

data = [gen_json(texto, pred) for
        texto, pred in zip(textos, preds)]

dump_set.import_data(data=data)






