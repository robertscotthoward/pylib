import os
import shutil
import pytest
from lib.ai.corpus import Corpus
from lib.ai.vectordb_chroma import ChromaVectorDb
from lib.ai.corpus import Corpus
from lib.ai.splitter import RecursiveCharacterText_Splitter


# INPUT
corpus_folder= os.path.abspath("data/corpus2")

# OUTPUT
collection_path= os.path.abspath("data/vdb/chroma/corpus2")


@pytest.fixture(scope="session", autouse=True)
def setup_teardown():
    print("\n[SETUP]")

    yield  # Call all the other tests
    

    print("\n[TEARDOWN]")


def test_vectordb_chroma_load_corpus():
    """
    Test the load_corpus method of the ChromaVectorDb class.
    """
    if os.path.exists(collection_path):
        # Remove this path and all its contents
        if not collection_path.startswith(os.path.abspath("")):
            raise ValueError(f"Collection path {collection_path} is not in the project root")
        shutil.rmtree(collection_path)
    corpus = Corpus(corpus_folder=corpus_folder)
    splitter = RecursiveCharacterText_Splitter(chunk_size=1000, chunk_overlap=200)
    vdb = ChromaVectorDb(corpus, splitter, collection_path=collection_path)
    vdb.load_corpus()
    assert vdb is not None
    assert vdb.collection_name == "corpus2"
    assert vdb.collection_path == collection_path
    assert vdb.collection.count() > 0 # Number of embeddings
    assert vdb.file_count_after == corpus.get_file_count()


def test_vectordb_chroma_query_corpus():
    vdb = ChromaVectorDb(collection_path=collection_path)
    assert vdb is not None
    assert vdb.collection_name == "corpus2"
    assert vdb.collection_path == collection_path
    assert vdb.collection.count() > 0 # Number of embeddings

    results = vdb.retrieve_documents("What did Butcher hear?")
    assert results is not None
    assert len(results) > 0
    assert "Kogloonian" in results



if __name__ == "__main__":
    # test_vectordb_chroma_load_corpus()
    test_vectordb_chroma_query_corpus()