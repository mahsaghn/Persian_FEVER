from drqa.retriever import DocDB, utils
import os


class FeverDocDB(DocDB):

    def __init__(self,path=None):
        #connect to database
        # arr = os.listdir()
        # print(arr)
        # print("______",path)
        super().__init__(path)

    def get_doc_lines(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT lines FROM documents WHERE id = ?",
            (utils.normalize(doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]

    def get_non_empty_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents WHERE length(trim(text)) > 0")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results