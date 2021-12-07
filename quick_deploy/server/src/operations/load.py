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
        new_split = []
        for i in range(len(split_result)):
            split_sentence = re.sub(' +', ' ', split_result[i])
            if len(split_sentence) > 0:
                new_split.append(split_sentence)
            # If split too short. Combine it with previous/next split.
            if len(split_sentence) < 7:
                index = i
                longer_sentence = split_sentence
                while index > 0 and len(longer_sentence) < 7:
                    index = index - 1
                    longer_sentence = split_result[index] + longer_sentence
                new_split.append(longer_sentence)
                index = i
                longer_sentence = split_sentence
                while index < len(split_result) - 1 and len(longer_sentence) < 7:
                    index = index + 1
                    longer_sentence = longer_sentence + split_result[index]
                new_split.append(longer_sentence)
        if include_original:
            new_split.append(sentence)
        result.append(new_split)
    return result


def process_data(sentence_list, model, is_title=False):
    doc_id = 0
    doc_id_list = []
    vector_list = []
    for sentence in sentence_list:
        doc_id = doc_id + 1
        doc_id_list.extend([doc_id] * len(sentence))
        vector_list.extend(model.sentence_encode(sentence))
    is_title_bit = 0
    if is_title:
        is_title_bit = 1
    return [doc_id_list, [is_title_bit] * len(doc_id_list), vector_list]


def create_search_data(file_dir, model):
    try:
        data = pd.read_csv(file_dir)
        title_data_raw = data['title'].tolist()
        text_data_raw = data['text'].tolist()
        title_data_split = do_split(title_data_raw, include_original=True)
        text_data_split = do_split(text_data_raw)
        insert_data = process_data(title_data_split, model, is_title=True)
        print('# of title data vectors = ', len(insert_data[0]))
        insert_text_data = process_data(text_data_split, model, is_title=False)
        print('# of text data vectors = ', len(insert_text_data[0]))
        for i in range(0, 3):
            insert_data[i].extend(insert_text_data[i])

        title_data_split.extend(text_data_split)
        data_split = []
        for sub_list in title_data_split:
            for item in sub_list:
                data_split.append(item)
        return title_data_raw, text_data_raw, insert_data, data_split
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


def format_sentence_data(insert_data, data_split):
    # Combine the id of the vector and the question data into a list
    data = []
    for i in range(len(insert_data[0])):
        value = (insert_data[0][i], insert_data[1][i], data_split[i])
        data.append(value)
    return data


# Import vectors to Milvus and data to Mysql respectively
def do_load(collection_name, file_dir, model, milvus_client, mysql_cli):
    if not collection_name:
        collection_name = DEFAULT_TABLE
    sentence_table = collection_name + '_sentence'

    title_data, text_data, insert_data, data_split = create_search_data(
        file_dir, model)
    # insert_data example:
    # [ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #   [1, 0, 1, 1, 0, 1, 0, 1, 1, 1],
    #   [
    #     [0.5104894743768512, 0.6560396299044534, 0.34525940750653883, 0.4907468233957538, 0.026554941012943756],
    #     [0.315488709152122, 0.5510008407021852, 0.6838450380157629, 0.22215404059303578, 0.5006650456587649],
    #     [0.8819106302308333, 0.004721144547834566, 0.7264288539867686, 0.8732102299341579, 0.7131052002167416],
    #     [0.1992932465527999, 0.6494109932323796, 0.38763433760343013, 0.3441007684528695, 0.23724898389646043],
    #     [0.4447237856781642, 0.07828555514365543, 0.25845219798269303, 0.9404709975001969, 0.3958210799455725]
    #   ]
    # ]
    ids = milvus_client.insert(collection_name, insert_data)
    milvus_client.create_index(collection_name)
    mysql_cli.create_mysql_text_table(collection_name)
    mysql_cli.create_mysql_sentence_table(sentence_table)
    mysql_cli.load_data_to_mysql(collection_name,
                                 format_data(title_data, text_data))
    insert_data[0] = ids
    mysql_cli.load_sentence_data_to_mysql(sentence_table,
                                          format_sentence_data(insert_data,
                                                               data_split))
    return len(ids)


if test:
    host = '127.0.0.1'
    port = '19530'
    connections.add_connection(default={"host": host, "port": port})
    connections.connect(alias='default')
    Collection('test_table').drop()
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
    print(collection.num_entities)
    milvus_ids = collection.insert(data).primary_keys

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
