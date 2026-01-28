import os
import shutil
import pytest
from lib.corpus import Corpus


# INPUT
corpus_folder= os.path.abspath("data/corpus1")

# OUTPUT
collection_path= os.path.abspath("data/vdb/chroma/corpus1")


@pytest.fixture(scope="session", autouse=True)
def initialize_everything():
    print("\n[SETUP]")
    if False and os.path.exists(collection_path):
        # Remove this path and all its contents
        if not collection_path.startswith(os.path.abspath("")):
            raise ValueError(f"Collection path {collection_path} is not in the project root")
        shutil.rmtree(collection_path)
    

    yield  # Call all the other tests
    

    print("\n[TEARDOWN]")


def test_vectordb_chroma():
    from lib.vectordb_chroma import ChromaVectorDb
    from lib.corpus import Corpus
    from lib.splitter import RecursiveCharacterText_Splitter

    corpus = Corpus(corpus_folder=corpus_folder)
    splitter = RecursiveCharacterText_Splitter(chunk_size=1000, chunk_overlap=200)
    vdb = ChromaVectorDb(corpus, splitter, collection_path=collection_path)
    if not os.path.exists(collection_path):
        vdb.load_corpus()
    assert vdb is not None
    assert vdb.collection_name == "corpus1"
    assert vdb.collection_path == collection_path
    assert vdb.collection.count() > 0 # Number of embeddings
    assert vdb.file_count_after == corpus.get_file_count()