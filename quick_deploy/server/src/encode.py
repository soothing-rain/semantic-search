import os
import zipfile

import gdown
import requests
from sentence_transformers import SentenceTransformer

from config import MODEL_PATH

test = False


class SentenceModel:
    """
    Say something about the ExampleCalass...

    Args:
        args_0 (`type`):
        ...
    """

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        if not os.path.exists("./paraphrase-mpnet-base-v2.zip"):
            url = 'https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/paraphrase-mpnet-base-v2.zip'
            gdown.download(url)
        with zipfile.ZipFile('paraphrase-mpnet-base-v2.zip', 'r') as zip_ref:
            zip_ref.extractall(MODEL_PATH)
        self.model = SentenceTransformer(MODEL_PATH)

    def sentence_encode(self, data):
        sentence_list = str(data).replace('\'', '\"')
        payload = "{\"sentences\":" + sentence_list + "}"
        headers = {
            'Content-type': 'application/json',
            'Accept': 'application/json'
        }
        r = requests.post('http://encoding-service:port/', headers=headers,
                          data=payload.encode('utf-8'))
        return r.json()['message']['vectors']
