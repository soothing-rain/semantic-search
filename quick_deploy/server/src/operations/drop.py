import sys
sys.path.append("..")
from config import DEFAULT_TABLE
from logs import LOGGER


def do_drop(table_name, milvus_cli, mysql_cli, is_sentence_table):
    if not table_name:
        if not is_sentence_table:
            table_name = DEFAULT_TABLE
        else:
            table_name = DEFAULT_TABLE + '_sentence'
    try:
        mysql_cli.delete_table(table_name)
        if not milvus_cli.has_collection(table_name):
            return "collection is not exist"
        status = milvus_cli.delete_collection(table_name)
        return status
    except Exception as e:
        LOGGER.error(f"Error with  drop table: {e}")
        sys.exit(1)
