import datetime as dt
import json
import pickle
import time

from ktrain import text as txt
from label_studio_sdk import Client
import ktrain

from utils import export_tasks_CONLL
from utils import gen_json
from utils import get_agreements
from utils import get_unlabeled_tasks
from utils import gen_predictors
from utils import get_unlabeled_tasks_ids
from utils import get_labeled_tasks
from utils import get_all_tasks
from utils import is_empty_project

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

modelos = [('BERTimbau-1', 'neuralmind/bert-base-portuguese-cased'),
           ('BERTimbau-2', 'neuralmind/bert-base-portuguese-cased'),
           ('BERTLeNERBR', 'pierreguillou/bert-base-cased-pt-lenerbr')]


time = dt.datetime.now()
start=True

while True:
    delta = dt.datetime.now() - time
    if delta.seconds >= 60 or start:

        start = False

        time = dt.datetime.now()

        # Pegamos apenas as tasks anotadas:
        print(f"[{dt.datetime.now()}] Procurando por novas anotações...")
        labeled_tasks = get_labeled_tasks(prediction_set)

        if(len(labeled_tasks)<=0 and not is_empty_project(prediction_set)):
            print("Não foram encrontradas novas anotações.")
            continue

        if len(labeled_tasks):
            ########################################################
            print(f"[{dt.datetime.now()}] Enviando anotações para projeto de treinamento...")
            # Pegando tasks do projeto de treinamento
            all_tasks = get_all_tasks(annotation_set)
            # Procuramos a task correspondente no projeto de treinamento 
            for i in range(len(all_tasks)):
                for j in range(len(labeled_tasks)):
                    if all_tasks[i]['id'] == labeled_tasks[j]['meta']['id']:
                        all_tasks[i]['annotations']=labeled_tasks[j]['annotations']
                        all_tasks[i]['is_labeled']=True
                        break

            # Deletamos anotações do projeto de treinamento
            annotation_set.make_request('DELETE', 'api/projects/39/tasks/')
            # importamos as novas anotações
            annotation_set.import_tasks(all_tasks)
            ########################################################

        print(f"[{dt.datetime.now()}] Baixando dados rotulados...")
        export_tasks_CONLL(annotation_set)

        print(f"[{dt.datetime.now()}] Treinando modelos...")
        predictors = gen_predictors(modelos, train_filepath=TDATA, val_filepath=VDATA)

        print(f"[{dt.datetime.now()}] Baixando dados não rotulados...")
        tasks_texts = get_unlabeled_tasks(annotation_set)
        unlabeled_ids = get_unlabeled_tasks_ids(annotation_set)

        print(f"[{dt.datetime.now()}] Realizando predições...")
        predictions = [predictor.predict(tasks_texts) for predictor in predictors]

        print(f"[{dt.datetime.now()}] Comparando predições...")
        agreements = get_agreements(tasks_texts, predictions, unlabeled_ids)
        tasks = [gen_json(**agreement) for agreement in agreements]

        print(f"[{dt.datetime.now()}] Atualizando predições...")
        prediction_set.make_request('DELETE', f'api/projects/{U_ID}/tasks/')
        prediction_set.import_tasks(tasks)

