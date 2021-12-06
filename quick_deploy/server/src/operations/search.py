import sys

sys.path.append("..")
from config import TOP_K, DEFAULT_TABLE
from logs import LOGGER
from pymilvus import Collection


def search_in_milvus(table_name, query_sentence, model, milvus_cli, mysql_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
        sentence_table = 'default_sentence'
    else:
        sentence_table = table_name + '_sentence'
    try:
        vectors = model.sentence_encode([query_sentence])
        LOGGER.info("Successfully insert query list")
        results = milvus_cli.search_vectors(table_name, vectors, TOP_K)
        vector_id_list = [x.id for x in results[0]]
        candidates = []
        for vector_id in vector_id_list:
            expr = "id == " + str(vector_id)
            query_res = Collection(name=table_name).query(expr, output_fields=[
                "doc_id", "is_title"])
            if len(query_res) > 0:
                candidates.append(query_res[0])
        target_doc_ids = []
        target_milvus_ids = []
        for milvus_hit in candidates:
            current_doc_id = milvus_hit['doc_id']
            if not (current_doc_id in target_doc_ids):
                target_doc_ids.append(current_doc_id)
                target_milvus_ids.append(str(milvus_hit['id']))
        ids, title, text = mysql_cli.search_by_doc_ids(target_doc_ids,
                                                       table_name)
        distances = [x.distance for x in results[0]]
        milvus_ids, is_title, sentence = mysql_cli.search_by_milvus_ids(
            target_milvus_ids, sentence_table)
        for i in range(len(text)):
            text[i] = "======[DEBUG info]Target Sentence: \"" + sentence[i] + \
                      "\" From title: " + str(is_title[i]) + "======" + text[i]
        return ids, title, text, distances
    except Exception as e:
        LOGGER.error(f" Error with search : {e}")
        sys.exit(1)
