import random
import re
import sys

import pandas as pd

sys.path.append("..")
from config import DEFAULT_TABLE
from logs import LOGGER
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema, \
    connections, list_collections

test = False


def do_split(data_list, include_original=False):
    result = []
    for sentence in data_list:
        sentence = sentence.replace('\n', ' ')
        sentence = sentence.replace('  ', ' ')
        split_result = re.split('[^a-zA-Z0-9_ 、“”%《》（）〈〉?\w\\-/]', sentence)
        if split_result[-1] == '':
            split_result.pop()
        if include_original:
            split_result.append(sentence)
        result.append(split_result)

    return result


def process_data(data_list, model, is_title=False):
    doc_id = 0
    doc_id_list = []
    vector_list = []
    for datum in data_list:
        doc_id = doc_id + 1
        doc_id_list.extend([doc_id] * len(datum))
        vector_list.extend(model.sentence_encode(datum))
    is_title_bit = 0
    if is_title:
        is_title_bit = 1
    return [doc_id_list, [is_title_bit] * len(doc_id_list), vector_list]


# Get the vector of question
# def extract_features(file_dir, model):
#     try:
#         data = pd.read_csv(file_dir)
#         title_data = data['title'].tolist()
#         text_data = data['text'].tolist()
#         sentence_embeddings = model.sentence_encode(title_data)
#         return title_data, text_data, sentence_embeddings
#     except Exception as e:
#         LOGGER.error(f" Error with extracting feature from question {e}")
#         sys.exit(1)


def create_search_data(file_dir, model):
    try:
        data = pd.read_csv(file_dir)
        title_data_raw = data['title'].tolist()
        text_data_raw = data['text'].tolist()
        title_data = do_split(title_data_raw, include_original=True)
        text_data = do_split(text_data_raw)
        insert_data = process_data(title_data, model, is_title=True)
        print('dimension = ', len(insert_data[0]), len(insert_data[1]), len(insert_data[2]))
        insert_text_data = process_data(text_data, model, is_title=False)
        print('dimension = ', len(insert_text_data[0]), len(insert_text_data[1]), len(insert_text_data[2]))
        for i in range(0, 3):
            insert_data[i].extend(insert_text_data[i])
        return title_data_raw, text_data_raw, insert_data
    except Exception as e:
        LOGGER.error(f" Error with extracting feature from question {e}")
        sys.exit(1)


def format_data(title_data, text_data):
    # Combine the id of the vector and the question data into a list
    data = []
    for i in range(len(title_data)):
        value = (i + 1, title_data[i], text_data[i])
        data.append(value)
    return data


# Import vectors to Milvus and data to Mysql respectively
def do_load(collection_name, file_dir, model, milvus_client, mysql_cli):
    if not collection_name:
        collection_name = DEFAULT_TABLE
    # title_data, text_data, sentence_embeddings = extract_features(file_dir,
    #                                                               model)

    title_data, text_data, insert_data = create_search_data(file_dir, model)
    ids = milvus_client.insert(collection_name, insert_data)
    milvus_client.create_index(collection_name)
    mysql_cli.create_mysql_table(collection_name)
    mysql_cli.load_data_to_mysql(collection_name, format_data(title_data, text_data))
    return len(ids)


if test:
    host = '127.0.0.1'
    port = '19530'
    connections.add_connection(default={"host": host, "port": port})
    connections.connect(alias='default')
    print(list_collections())
    dim = 5
    field1 = FieldSchema(name="id", dtype=DataType.INT64,
                         descrition="int64")
    field2 = FieldSchema(name="doc_id",
                         dtype=DataType.INT64,
                         description="document id")
    field3 = FieldSchema(name="is_title",
                         dtype=DataType.INT64,
                         description="if embedding is document title")
    field4 = FieldSchema(name="embedding",
                         dtype=DataType.FLOAT_VECTOR,
                         descrition="float vector",
                         dim=dim)
    schema = CollectionSchema(fields=[field1, field2, field3, field4],
                              primary_field='id',
                              auto_id=True,
                              description="collection description")
    collection_name = "tutorial"
    collection = Collection(name=collection_name, schema=schema)
    print(collection.partitions)

    data = [
        [i for i in range(10)],
        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [[random.random() for _ in range(5)] for _ in range(10)],
    ]
    print("INSERTING...", data)
    collection.insert(data)
    print(collection.num_entities)

    query_embedding = [[random.random() for _ in range(5)] for _ in range(1)]
    anns_field = "embedding"
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    limit = 5
    expr = "is_title == 1"
    collection.load()
    print("SEARCHING...", query_embedding)
    results = collection.search(query_embedding, anns_field, search_params,
                                limit, expr)

    print(len(results), results)
    print('type of results[0]', type(results[0]))
    print('id list = ', [str(x.id) for x in results[0]])
    id_list = [x.id for x in results[0]]

    result = []
    for id in id_list:
        expr = "id == " + str(id)
        print('expr = ', expr)
        query_res = collection.query(expr, output_fields=["doc_id", "is_title"])
        if len(query_res) > 0:
            result.append(query_res[0])

    print('result = ', result)
    doc_ids = []
    for item in result:
        doc_candi = item['doc_id']
        if not (doc_candi in doc_ids):
            doc_ids.append(doc_candi)
    print('doc_ids list', doc_ids)
    collection.drop()
