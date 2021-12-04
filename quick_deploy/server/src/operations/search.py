import sys
import numpy as np

sys.path.append("..")
from config import TOP_K, DEFAULT_TABLE
from logs import LOGGER
from pymilvus import Collection


def search_in_milvus(table_name, query_sentence, model, milvus_cli, mysql_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        vectors = model.sentence_encode([query_sentence])
        LOGGER.info("Successfully insert query list")
        results = milvus_cli.search_vectors(table_name, vectors, TOP_K)
        id_list = [x.id for x in results[0]]
        candidates = []
        for id in id_list:
            expr = "id == " + str(id)
            query_res = Collection(name=table_name).query(expr, output_fields=["doc_id", "is_title"])
            if len(query_res) > 0:
                candidates.append(query_res[0])
        doc_ids = []
        for item in candidates:
            doc_candi = item['doc_id']
            if not (doc_candi in doc_ids):
                doc_ids.append(doc_candi)
        print("----------------\ntarget doc ids = ", doc_ids)
        ids, title, text = mysql_cli.search_by_doc_ids(doc_ids, table_name)
        distances = [x.distance for x in results[0]]
        return ids, title, text, distances
    except Exception as e:
        LOGGER.error(f" Error with search : {e}")
        sys.exit(1)
