import json
import pickle
import time

# import ktrain
# from ktrain import text as txt

from label_studio_sdk import Client

from utils import gen_json, export_tasks_CONLL, export_tasks_text

LABEL_STUDIO_URL = 'http://164.41.76.30/labelstudio'
API_KEY =  'bc36020e5d03487292cac63d82661daa12320042'

L_ID = '39'
U_ID = '40'

TDATA = 'train.conll'
VDATA = 'train.conll'

ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
ls.check_connection()


labeled_set   = ls.get_project(L_ID)
unlabeled_set = ls.get_project(U_ID)

# print("Downloading training data from Label Studio...")
# tic = time.perf_counter()
# export_tasks_CONLL(labeled_set)
# toc = time.perf_counter()
# print(f'Done![{toc - tic:0.4f}s]')

# (trn, val, preproc) = txt.entities_from_conll2003(TDATA,
#                                                   val_filepath=VDATA,
#                                                   verbose=0)

# print("Criando modelo")
# tic = time.perf_counter()
# model_bertimbau = txt.sequence_tagger(
#     'bilstm-bert', preproc, verbose=0,
#     transformer_model='neuralmind/bert-base-portuguese-cased')
# toc = time.perf_counter()
# print(f'Done![{toc - tic:0.4f}s]')

# print("Criando learner")
# tic = time.perf_counter()
# learner_bertimbau = ktrain.get_learner(
#     model_bertimbau, train_data=trn,
#     val_data=val, batch_size=128)
# toc = time.perf_counter()
# print(f'Done![{toc - tic:0.4f}s]')

# print("Treinando learner")
# tic = time.perf_counter()
# learner_bertimbau.fit(
#     0.01, 1, cycle_len=5,
#     checkpoint_folder='/tmp/saved_weights_bertimbau')
# toc = time.perf_counter()
# print(f'Done![{toc - tic:0.4f}s]')

# predictor_bertimbau = ktrain.get_predictor(
#     learner_bertimbau.model, preproc)


texts = export_tasks_text(unlabeled_set)
print(texts)

# data = [gen_json(texto, pred) for texto, pred in zip(textos, preds)]

# dump_set.import_data(data=data)






