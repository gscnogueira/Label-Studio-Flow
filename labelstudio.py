import requests
import json

class Project:

    def __init__(self, server, project_id, token):
        self.server = server
        self.project_id = project_id
        self.token = token

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

    def test(self):

        response = requests.get(f'http://{self.server}/labelstudio/api/projects/{self.project_id}/export/formats',
                                 headers={'Authorization': f'Token {self.token}'},
                                 )

        print(response.text)



