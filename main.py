import argparse
import configparser
import datetime as dt
import json
from os.path import exists
import pickle
import time

from labelstudio import ls_Client
from model_query import Models


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

ls = ls_Client(server=LABEL_STUDIO_URL,token=API_KEY)
model_query = Models()

annotation_set = ls.client.get_project(L_ID)
prediction_set = ls.client.get_project(U_ID)


time = dt.datetime.now()
start=True

while True:
    delta = dt.datetime.now() - time
    if delta.seconds >= 60 or start:

        start = False

        time = dt.datetime.now()

        # Pegamos apenas as tasks anotadas:
        print(f"[{dt.datetime.now()}] Procurando por novas anotações...")
        labeled_tasks = ls.get_labeled_tasks(project=prediction_set)

        if(len(labeled_tasks)<=0 and not ls.is_empty_project(prediction_set)):
            print("Não foram encrontradas novas anotações.")
            continue

        if len(labeled_tasks):
            print(f"[{dt.datetime.now()}] Enviando anotações para projeto de treinamento...")
            # transfere anotações de um projeto para o outro
            ann = ls.transfer_annotations(ls.client,labeled_tasks)
            with open('temp1.json', 'w') as f:
                json.dump(ann, f)

        print(f"[{dt.datetime.now()}] Baixando dados rotulados...")
        ls.export_tasks_CONLL(annotation_set)

        print(f"[{dt.datetime.now()}] Treinando modelos...")
        predictors = model_query.gen_predictors(models, train_filepath=TDATA, val_filepath=VDATA)

        print(f"[{dt.datetime.now()}] Baixando dados não rotulados...")
        tasks_texts = ls.get_unlabeled_tasks(annotation_set)
        unlabeled_ids = ls.get_unlabeled_tasks_ids(annotation_set)

        print(f"[{dt.datetime.now()}] Realizando predições...")
        predictions = [predictor.predict(tasks_texts) for predictor in predictors]

        print(f"[{dt.datetime.now()}] Comparando predições...")
        agreements = model_query.get_agreements(tasks_texts, predictions, unlabeled_ids)
        tasks = [ls.gen_json(**agreement) for agreement in agreements]

        print(f"[{dt.datetime.now()}] Atualizando predições...")

        ########################################################
        # Update preventivo para manter predições realizadas durante o tempo de treinamento dos modelos
        labeled_tasks_new = ls.get_labeled_tasks(prediction_set)
        ########################################################

        prediction_set.make_request('DELETE', f'api/projects/{U_ID}/tasks/')
        prediction_set.import_tasks(tasks)

