import re

def get_result(texto, pred):

    start = 0
    end   = 0
    label = None

    result = []

    while len(pred)>0 and end < len(texto):

        
        pattern = re.compile('[\s\-]')
        if re.match(pattern, texto[end]):
            end+=1
            continue

        token, tag  = pred.pop(0)

        if tag[0] != 'I' and end>0:
            if label:
                result.append({"id": str(end),
                               "from_name": "label",
                               "to_name": "text",
                               "type":"labels",
                               # "readonly": False,
                               "value":{
                                   'start':start,
                                   'end' : end,
                                   'text' : texto[start:end],
                                   'score': 0.5,
                                   'labels' : [label]
                               }
                               })
            start = end

        end+=len(token)
        label = tag[2:] if label != 'O' else ''
    return result

def gen_json(texto, pred):

    json_ = {"data":{} , "predictions":[]}
    json_["data"]["text"] = texto

    json_["predictions"].append({"model_version": 'BERTimbau',
                                 "result": get_result(texto, pred)})
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


