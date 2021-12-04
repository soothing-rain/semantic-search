from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from config import MODEL_PATH
import gdown
import zipfile
import os

import pandas as pd
import re

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
        embedding = self.model.encode(data)
        sentence_embeddings = normalize(embedding)
        return sentence_embeddings.tolist()

def do_split(data_list, include_original=False):
    result = []
    for sentence in data_list:
        sentence = sentence.replace('\n', ' ').replace('  ', ' ')
        split_result = re.split('[^a-zA-Z0-9_ 、“”%《》（）〈〉?\w\\-/]', sentence)
        if split_result[-1] == '':
            split_result.pop()
        if include_original:
            split_result.append(sentence)
        result.append(split_result)

    return result


def process_data(data_list, is_title=False):
    model = SentenceModel()
    doc_id = 0
    doc_id_list = []
    vector_list = []
    for datum in data_list:
        doc_id = doc_id+1
        doc_id_list.extend([doc_id] * len(datum))
        vector_list.extend(model.sentence_encode(datum))
    return [doc_id_list, [is_title] * len(doc_id_list), vector_list]


if test:
    data = pd.read_csv("../../../data/testdata.csv")

    title_data = do_split(data['title'].tolist(), include_original=True)
    text_data = do_split(data['text'].tolist())

    print(title_data)
    print(text_data)

    insert_data = process_data(title_data, is_title=True)
    #print(insert_data)

    insert_data_2 = process_data(text_data, is_title=False)
    insert_data[0].extend(insert_data_2[0])
    insert_data[1].extend(insert_data_2[1])
    insert_data[2].extend(insert_data_2[2])
    print(insert_data[0], insert_data[1], len(insert_data[2]))
    print(max(insert_data[0]))
