import argparse
import configparser
import datetime as dt
import json
from os.path import exists
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
from utils import is_empty_project
from utils import transfer_annotations

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("config_file", help="path to configuration file")
args = vars(parser.parse_args())
config_file = args['config_file'] 

if not exists(config_file):
    print('ERROR: File does not exist')
    exit(1)

config = configparser.ConfigParser()
config.read(config_file)
user_config = config['USER']
model_config = config['MODELS']

LABEL_STUDIO_URL = user_config['label_studio_url']
API_KEY =  user_config['label_studio_api_key']
L_ID = user_config['labeled_project_id']
U_ID = user_config['unlabeled_project_id']
TDATA = user_config.get('train_path')

models = [o for o in config.items('MODELS')
          if o not in config.defaults().items()]

ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
ls.check_connection()

annotation_set = ls.get_project(L_ID)
prediction_set = ls.get_project(U_ID)

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
            print(f"[{dt.datetime.now()}] Enviando anotações para projeto de treinamento...")
            # transfere anotações de um projeto para o outro
            transfer_annotations(ls,labeled_tasks)

        print(f"[{dt.datetime.now()}] Baixando dados rotulados...")
        export_tasks_CONLL(annotation_set)

        print(f"[{dt.datetime.now()}] Treinando modelos...")
        predictors = gen_predictors(models, train_filepath=TDATA, val_filepath=TDATA)

        print(f"[{dt.datetime.now()}] Baixando dados não rotulados...")
        tasks_texts = get_unlabeled_tasks(annotation_set)
        unlabeled_ids = get_unlabeled_tasks_ids(annotation_set)

        print(f"[{dt.datetime.now()}] Realizando predições...")
        predictions = [predictor.predict(tasks_texts) for predictor in predictors]

        print(f"[{dt.datetime.now()}] Comparando predições...")
        agreements = get_agreements(tasks_texts, predictions, unlabeled_ids)
        tasks = [gen_json(**agreement) for agreement in agreements]

        print(f"[{dt.datetime.now()}] Atualizando predições...")
        ########################################################
        # Update preventivo para manter predições realizadas durante o tempo de treinamento dos modelos
        labeled_tasks_new = get_labeled_tasks(prediction_set)
        # para evitar a adição de anotações repetidas
        for i in range(len(labeled_tasks)):
            for j in range(len(labeled_tasks_new)):
                if(labeled_tasks_new[j]['id']==labeled_tasks[i]['id']):
                    labeled_tasks_new.pop(j)
                    break
        # transfere anotações de um projeto para o outro
        transfer_annotations(ls,labeled_tasks_new)
        ########################################################
        prediction_set.make_request('DELETE', f'api/projects/{U_ID}/tasks/')
        prediction_set.import_tasks(tasks)

