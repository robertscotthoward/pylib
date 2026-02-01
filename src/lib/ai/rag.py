from lib.ai.modelstack import *
from lib.tools import *




class Rag:
    def __init__(self, vdb, llm):
        # collection_name is the name of the collection to use
        # corpus_folder is the folder containing the corpus to use
        # collection_folder is where the vector database is stored
        # model_config is the configuration for the model to use
        self.collection_name = vdb.collection_name
        self.corpus_folder = vdb.corpus.corpus_folder
        self.llm = llm
        self.vdb = vdb

    def get_document_count(self):
        """Return the number of unique documents in this RAG."""
        try:
            print(f"[DEBUG] Rag.get_document_count: collection_name={self.collection_name}")
            count = self.vdb.get_file_count()
            print(f"[DEBUG] Rag.get_document_count: returning {count}")
            return count
        except Exception as e:
            print(f"[ERROR] Rag.get_document_count: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def get_entry_count(self):
        """Return the number of entries (chunks) in this RAG."""
        try:
            print(f"[DEBUG] Rag.get_entry_count: collection_name={self.collection_name}")
            count = self.vdb.get_entry_count()
            print(f"[DEBUG] Rag.get_entry_count: returning {count}")
            return count
        except Exception as e:
            print(f"[ERROR] Rag.get_entry_count: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def sync_corpus_to_vdb(self):
        """
        Sync corpus to vector database: load any documents that don't exist in the VDB.
        Returns a tuple of (files_added, chunks_added).
        """
        try:
            print(f"[DEBUG] Rag.sync_corpus_to_vdb: collection_name={self.collection_name}")
            
            # Get list of files already embedded in the VDB
            embedded_files = self.vdb.get_embedded_files()
            embedded_filenames = set([row[0] for row in embedded_files])
            
            print(f"[DEBUG] Already embedded: {len(embedded_filenames)} files")
            
            # Get list of all files in corpus
            from lib.ai.corpus import Corpus
            corpus = Corpus()
            all_files = list(corpus.enumerate_files(self.corpus_folder))
            
            print(f"[DEBUG] Total files in corpus: {len(all_files)}")
            
            # Find files that need to be added
            files_to_add = [f for f in all_files if f not in embedded_filenames]
            
            print(f"[DEBUG] Files to add: {len(files_to_add)}")
            
            # Add missing files to VDB
            total_chunks = 0
            for filepath in files_to_add:
                n_chunks = self.vdb.add_document(filepath)
                if n_chunks:
                    total_chunks += n_chunks
                    print(f"[DEBUG] Added {filepath}: {n_chunks} chunks")
            
            # Commit any remaining batch
            self.vdb.commit_batch()
            
            # Save the index
            self.vdb.save_database()
            
            print(f"[DEBUG] Sync complete: {len(files_to_add)} files added, {total_chunks} chunks total")
            return (len(files_to_add), total_chunks)
            
        except Exception as e:
            print(f"[ERROR] Rag.sync_corpus_to_vdb: {e}")
            import traceback
            traceback.print_exc()
            return (0, 0)

    def query(self, query):
        nResults =  int(self.llm.num_tokens() / (1000/5))
        context = self.vdb.retrieve_documents(query, n_results=nResults)
        prompt = f"""
QUERY: {query}

CONTEXT: {context}
        """
        answer = self.llm.query(prompt)
        return answer





if __name__ == "__main__":
    from lib.ai.vectordb_common import make_vectordb
    collection_name = "corpus1"
    vdb_type = "numpy"
    corpus_folder = os.path.abspath(fr"C:\Rob\GitHub\robertscotthoward\python-ollama-example\data\test\{collection_name}")
    collection_folder = os.path.abspath(f"data/vdb/{vdb_type}/{collection_name}")
    vdb = make_vectordb(collection_name, corpus_folder, collection_folder, load_corpus=True)
    config = {
        'class': 'ollama',
        'host': 'http://localhost:11434',
        'model': 'granite3.2:2b',
        'context-window': '128K'
    }
    rag = Rag(vdb, corpus_folder, collection_folder, config)

    queries = [
        "I want to shift into another plane. What spell should I use?",
        "Who did Howard know?",
    ]
    for query in queries:
        print(f"Query: {query}")
        answer = rag.query(query)
        print(f"Answer: {answer}")
        print("-" * 80)


