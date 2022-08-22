import requests
import json
import re
from label_studio_sdk import Client

class Project():

    def __init__(self, project_id, server, token):
        #super().__init__(ls, server, token)
        self.server = server
        self.token = token
        self.project_id = project_id

    def export_tasks(self, export_type="CONLL2003", file='train.conll', snapshot=None):

        response = requests.get(f'http://{self.server}/labelstudio/api/projects/{self.project_id}/exports/{snapshot}/download',
                                headers={'Authorization': f'Token {self.token}'},
                                params={'exportType' : export_type})

        f = open(file, 'w')
        f.write(response.text)
        f.close()

    def get_json(self):

        response = requests.get(f'http://{self.server}/labelstudio/api/projects/{self.project_id}/exports/3/download',
                                headers={'Authorization': f'Token {self.token}'})

        return response.json()

    def get_txt(self):

        texts = [task['data']['text'] for task in self.get_json()]
        return texts

    def snapshot(self):

        response = requests.post(f'http://{self.server}/labelstudio/api/projects/{self.project_id}/exports/',
                                headers={'Authorization': f'Token {self.token}'})
        return response.text

    def list_snapshots(self):

        response = requests.get(f'http://{self.server}/labelstudio/api/projects/{self.project_id}/exports/',
                                headers={'Authorization': f'Token {self.token}'})
        return response.text

    def import_tasks(self, file=None):

        response = requests.post(f'http://{self.server}/labelstudio/api/projects/{self.project_id}/import',
                                 headers={'Authorization': f'Token {self.token}'},
                                 files = {'blau':open(file, 'rb')})

        print(response.text)

    def import_data(self, data=None):

        response = requests.post(f'http://{self.server}/labelstudio/api/projects/{self.project_id}/import',
                                 headers={'Authorization': f'Token {self.token}'},
                                 json = data)
        print(response.text)

    # def test(self):

    #     response = requests.get(f'http://{self.server}/labelstudio/api/projects/{self.project_id}/export/formats',
    #                              headers={'Authorization': f'Token {self.token}'},
    #                              )

    #     print(response.text)



#class Utils:
class ls_Client:
    def __init__(self, server, token):
        self.server = server
        self.token = token
        self.client = Client(url=server, api_key=token)
        self.client.check_connection()

    def gen_annotation(self, text, start, end, label):
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

    def get_result(self, texto, pred):

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
                    result.append(self.gen_annotation(text=texto,
                                                start=start,
                                                end=end,
                                                label=prev_label))
                start = end

            end += len(token)
            prev_label=label

        if prev_label!='O':
            result.append(self.gen_annotation(text=texto,
                                        start=start,
                                        end=end,
                                        label=prev_label))
        return result

    def gen_json(self, text, prediction, model_version, id):

        json_ = {"meta": {}, "data":{} , "predictions":[]}
        json_["meta"]["id"] = id
        json_["data"]["text"] = text

        json_["predictions"].append({"model_version": model_version,
                                    "result": self.get_result(text, prediction),
                                    "score":0.9})
        return json_

    def export_tasks_CONLL(self, project):
        export_type="CONLL2003"
        response = project.make_request(
            method='GET',
            url=f'/api/projects/{project.id}/export?exportType={export_type}')
        with open('train.conll', 'w') as f:
            f.write(response.text)

    def export_tasks_text(self, project):
        response = project.make_request('get',
                                        f'/api/projects/{project.id}/tasks',
                                        {'page_size': -1})
        tasks = response.json()
        return [task['data']['text'] for task in tasks]


    def get_tasks_ids(self, project):
        response = project.make_request('get',
                                        f'/api/projects/{project.id}/tasks',
                                        {'page_size': -1})
        tasks = response.json()
        return [task['id']  for task in tasks]

    def get_unlabeled_tasks(self, project):
        response = project.make_request('get',
                                        f'/api/projects/{project.id}/tasks',
                                        {'page_size': -1})
        tasks = response.json()
        return [task['data']['text'] for task in tasks if not task['annotations']]

    def get_labeled_tasks(self, project):
        response = project.make_request('get',
                                        f'/api/projects/{project.id}/tasks',
                                        {'page_size': -1})
        tasks = response.json()
        return [task for task in tasks if task['is_labeled'] == True]

    def get_all_tasks(self, project):
        response = project.make_request('get',
                                        f'/api/projects/{project.id}/tasks',
                                        {'page_size': -1})
        tasks = response.json()
        return tasks

    def get_unlabeled_tasks_ids(self, project):
        response = project.make_request('get',
                                        f'/api/projects/{project.id}/tasks',
                                        {'page_size': -1})
        tasks = response.json()
        return [task['id'] for task in tasks if not task['annotations']]

    def is_empty_project(self, project):
        response = project.make_request('get',
                                        f'/api/projects/{project.id}/tasks',
                                        {'page_size': 1})
        tasks = response.json()
        return len(tasks)==0

    def transfer_annotations(self, ls,labeled_tasks):
        annotations = []
        for labeled_task in labeled_tasks:
            task_id = labeled_task['meta']['id']
            for ann in labeled_task['annotations']:
                annotation = {"result": ann['result'],
                            "was_cancelled":ann['was_cancelled'],
                            "ground_truth":ann['ground_truth'],
                            "lead_time":ann['lead_time'],
                            "task":task_id, 
                            "completed_by":ann['completed_by']
                            } 

                get=ls.make_request('get',
                                    f'/api/tasks/{task_id}/annotations/')

                list_old_ann = get.json()
                print(list_old_ann)

                if(len(list_old_ann)<=0):
                    response=ls.make_request('post',
                                    f'/api/tasks/{task_id}/annotations/',
                                    json=annotation)
                else:
                    for ann_get in list_old_ann:
                        id_ann_get = ann_get['id']
                        ls.make_request('delete',
                                    f'/api/annotations/{id_ann_get}/')

                    response=ls.make_request('post',
                                    f'/api/tasks/{task_id}/annotations/',
                                    json=annotation)

                annotations.append(response.json())
        return annotations