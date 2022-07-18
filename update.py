import datetime as dt
import json
import pickle
import time

import ktrain
from ktrain import text as txt

from label_studio_sdk import Client

import utils
from utils import export_tasks_CONLL
from utils import gen_json
from utils import get_agreements
from utils import get_entities_from_prediction
from utils import get_result
from utils import get_unlabeled_tasks
from utils import get_labeled_tasks
from utils import get_all_tasks
from utils import train_model

LABEL_STUDIO_URL = 'http://164.41.76.30/labelstudio'
API_KEY =  'bc36020e5d03487292cac63d82661daa12320042'

L_ID = '39'
U_ID = '40'

TDATA = 'train.conll'
VDATA = 'train.conll'

ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
ls.check_connection()

annotation_set = ls.get_project(L_ID)
prediction_set = ls.get_project(U_ID)


# Usando o date time para testar a existencia de novas anotações a cada minuto
time = dt.datetime.now()

while True:
    delta = dt.datetime.now() - time
    if delta.seconds >= 60:
        print("Procurando por novas anotações")
        # Pegamos apenas as tasks anotadas:
        labeled_tasks = get_labeled_tasks(prediction_set)
        if(len(labeled_tasks)>0):
            print("Enviando anotações para projeto de treinamento")
            # Pegando tasks do projeto de treinamento
            old_tasks = get_all_tasks(annotation_set)
            # Concatenamos as tasks
            all_tasks = old_tasks + labeled_tasks
            # Deletamos anotações
            annotation_set.make_request('DELETE', 'api/projects/40/tasks/')
            # importamos as novas anotações
            annotation_set.import_tasks(all_tasks)


            # Aqui chamamariamos uma funcao para o main.py:
            # train_models()

            # OU....
            # Executavamos o treinamento de lá:

            # (trn, val, preproc) = txt.entities_from_conll2003(TDATA,
            #                                             val_filepath=VDATA,
            #                                             verbose=0)

            # predictors = []
            # for (model, source) in modelos :
            #     predictors.append(train_model(name=model,
            #                                 transformer_model=source,
            #                                 trn=trn, val=val,
            #                                 preproc=preproc))

            # print("Downloading unlabeled tasks ...")
            # tasks_texts = get_unlabeled_tasks(annotation_set)

            # print("Realizando predições ...")
            # predictions = [predictor.predict(tasks_texts) for predictor in predictors]

            # print("Comparando predições ...")
            # agreements = get_agreements(tasks_texts, predictions )
            # tasks = [gen_json(**agreement) for agreement in agreements]

            # print("Atualizando predições ...")
            # prediction_set.make_request('DELETE', 'api/projects/40/tasks/')
            # prediction_set.import_tasks(tasks)

    
        else:
            print("Não foram encrontradas novas anotações")

        time = dt.datetime.now()
