import os
import pytest
from lib.corpus import Corpus


corpus = Corpus(corpus_folder="data/corpus1")
collection_path= os.path.abspath("data/vdb/chroma/corpus1")

@pytest.fixture(scope="session", autouse=True)
def initialize_everything():
    # --- SETUP CODE ---
    print("\n[INIT] Connecting to database / Setting up environment")
    
    yield  # This is where the tests actually run
    
    # --- TEARDOWN CODE (Optional) ---
    print("\n[CLEANUP] Closing connections")


def test_vectordb_chroma():
    from lib.vectordb_chroma import ChromaVectorDb
    from lib.corpus import Corpus
    from lib.splitter import RecursiveCharacterText_Splitter

    splitter = RecursiveCharacterText_Splitter(chunk_size=1000, chunk_overlap=200)
    vdb = ChromaVectorDb(corpus, splitter, collection_path="data/vdb/chroma/corpus1")
    assert vdb is not None
    assert vdb.collection_name == "corpus1"
    assert vdb.collection_path == "data/vdb/chroma/corpus1"
    assert vdb.collection.count() > 0
    assert vdb.collection.count() == corpus.file_count