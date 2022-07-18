import re

import ktrain 
from ktrain import text as txt

def train_model(name, transformer_model, trn, val, preproc):

    print(name.center(50, '-'))

    model= txt.sequence_tagger('bilstm-bert',
                               preproc, verbose=0,
                               transformer_model=transformer_model)

    learner = ktrain.get_learner(model,
                                 train_data=trn,
                                 val_data=val,
                                 batch_size=128)

    learner.fit(0.01, 1, cycle_len=5,
                checkpoint_folder=f'/tmp/saved_weights_{name}')

    predictor = ktrain.get_predictor(learner.model, preproc)

    return predictor

def get_agreements(texts,predictions ) :

    agreements =[]
    entities = [get_entities_from_prediction(prediction) for prediction in predictions]

    for i in range(len(texts)):
        veredicts = {}
        majority    = 0;
        majority_id = 0;

        for j in range(len(predictions)):
            veredicts[j]=0
            for k in range(len(predictions)):
                if(entities[j][i] == entities[k][i]):
                    veredicts[j]+=1

        for v in veredicts:        
            if veredicts[v] > majority :
                majority = veredicts[v]
                majority_id = v

        if majority > len(predictions)//2:
            agreements.append({'text':texts[i],
                               'prediction': predictions[majority_id][i],
                               'model_version':'concordancia'})

    return agreements

def get_entities_from_prediction(predictions):

    predicted_entities = []

    for pred in predictions:
        entities = {}
        entity = []
        prev_label = 'O'
        for token, iob in pred:
            label = iob[2:] if len(iob) > 2 else 'O'
            is_begin = (iob[0] == 'B')

            if label!=prev_label or is_begin:
                if prev_label!='O':
                    entities[prev_label] = " ".join(entity)
                    entity = []

            entity.append(token)
            prev_label = label
        predicted_entities.append(entities)

    return predicted_entities

def gen_annotation(text, start, end, label):
    return {"id": str(end),
            "from_name": "label",
            "to_name": "text",
            "type":"labels",
            "value":{
                'start':start,
                'end' : end,
                'text' : text[start:end],
                'score': 0.5,
                'labels' : [label]
            }}

def get_result(texto, pred):

    start = 0
    end   = 0
    result = []
    prev_label='O'

    while len(pred)>0 and end < len(texto):

        pattern = re.compile('[\s\-]')
        if re.match(pattern, texto[end]):
            end+=1
            continue

        token, tag  = pred.pop(0)
        label = tag[2:] if tag != 'O' else 'O'
        pos = tag[0]


        if (label!=prev_label or pos == 'B'):
            if prev_label!='O':
                result.append(gen_annotation(text=texto,
                                             start=start,
                                             end=end,
                                             label=prev_label))
            start = end

        end += len(token)
        prev_label=label

    if prev_label!='O':
        result.append(gen_annotation(text=texto,
                                     start=start,
                                     end=end,
                                     label=prev_label))
    return result

def gen_json(text, prediction, model_version):

    json_ = {"data":{} , "predictions":[]}
    json_["data"]["text"] = text

    json_["predictions"].append({"model_version": model_version,
                                 "result": get_result(text, prediction),
                                 "score":0.9})
    return json_

def export_tasks_CONLL(project):
    export_type="CONLL2003"
    response = project.make_request(
        method='GET',
        url=f'/api/projects/{project.id}/export?exportType={export_type}')
    with open('train.conll', 'w') as f:
        f.write(response.text)

def export_tasks_text(project):
    response = project.make_request('get',
                                    f'/api/projects/{project.id}/tasks',
                                    {'page_size': -1})
    tasks = response.json()
    return [task['data']['text'] for task in tasks]


def get_tasks_ids(project):
    response = project.make_request('get',
                                    f'/api/projects/{project.id}/tasks',
                                    {'page_size': -1})
    tasks = response.json()
    return [task['id']  for task in tasks]

def get_unlabeled_tasks(project):
    response = project.make_request('get',
                                    f'/api/projects/{project.id}/tasks',
                                    {'page_size': -1})
    tasks = response.json()
    return [task['data']['text'] for task in tasks if not task['annotations']]

# Adicionei essas duas funcoes:
def get_labeled_tasks(project):
    response = project.make_request('get',
                                    f'/api/projects/{project.id}/tasks',
                                    {'page_size': -1})
    tasks = response.json()
    return [task for task in tasks if task['is_labeled'] == True]

def get_all_tasks(project):
    response = project.make_request('get',
                                    f'/api/projects/{project.id}/tasks',
                                    {'page_size': -1})
    tasks = response.json()
    return tasks


