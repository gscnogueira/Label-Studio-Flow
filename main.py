import json
import pickle
import time

import ktrain
from ktrain import text as txt

from label_studio_sdk import Client

import utils
from utils import get_result
from utils import gen_json
from utils import export_tasks_CONLL
from utils import export_tasks_text
from utils import train_model
from utils import get_entities_from_prediction

LABEL_STUDIO_URL = 'http://164.41.76.30/labelstudio'
API_KEY =  'bc36020e5d03487292cac63d82661daa12320042'

L_ID = '39'
U_ID = '40'

TDATA = 'train.conll'
VDATA = 'train.conll'

ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
ls.check_connection()


annotation_set   = ls.get_project(L_ID)
prediction_set = ls.get_project(U_ID)

print("Downloading training data from Label Studio...")
export_tasks_CONLL(annotation_set)

modelos = [('BERTimbau-1', 'neuralmind/bert-base-portuguese-cased'),
           ('BERTimbau-2', 'neuralmind/bert-base-portuguese-cased'),
           ('BERTLeNERBR', 'pierreguillou/bert-base-cased-pt-lenerbr')]

predictors = []

(trn, val, preproc) = txt.entities_from_conll2003(TDATA,
                                                  val_filepath=VDATA,
                                                  verbose=0)

for (model, source) in modelos :
    predictors.append(train_model(name=model,
                                  transformer_model=source,
                                  trn=trn, val=val,
                                  preproc=preproc))

print("Downloading Tasks ...")
tasks_texts = export_tasks_text(anotation_set)

print("Realizando predições ...")
predictions = [predictor.predict(tasks_texts) for predictor in predictors]

print("Comparando predições ...")
agreements = get_agreements(task_texts, predictions )
tasks = [gen_json(**agreement) for agreement in agreements]

print("Atualizando predições ...")
prediction_set.make_request('DELETE', 'api/projects/40/tasks/')
prediction_set.import_tasks(tasks)

