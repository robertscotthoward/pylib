import os
import json
import datetime
import re
import sqlite3
import chromadb
from lib.ai.fileconvert import get_text
from lib.ai.splitter import *
from lib.tools import *
from lib.ai.corpus import *
from chromadb.config import Settings
from lib.ai.vectordb_chroma import ChromaVectorDb
from lib.ai.vectordb_numpy import NumpyVectorDb



def make_vectordb(collection_name, corpus_folder, collection_folder=None, reranker=None, load_corpus=True, model_name="all-MiniLM-L6-v2"):
    corpus = Corpus(corpus_folder=corpus_folder)
    if load_corpus and corpus_folder:
        corpus.convert_files()
    collection_folder = collection_folder or os.path.abspath(f"data/vdb/{collection_name}")

    splitter = RecursiveCharacterText_Splitter(chunk_size=1000, chunk_overlap=200)
    vdb = ChromaVectorDb(corpus, splitter, collection_path=collection_folder, reranker=reranker, model_name=model_name)

    # Store cache file in the same directory as the vector database
    cache_file_path = os.path.join(collection_folder, "chroma_rag.json")
    
    # Read cache file (using direct file I/O to handle absolute paths)
    cache = {}
    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        except:
            pass
    
    collections = cache.get("collections", {})
    collection = collections.get(collection_name, {})
    last_updated = collection.get("last_updated", 0)
    if load_corpus and corpus_folder:
        last_updated = vdb.load_corpus(last_updated)
    else:
        last_updated = collection.get("last_updated", 0)
        vdb.file_count_after = 0
    collection["last_updated"] = last_updated
    cache["collections"] = collections
    collections[collection_name] = collection
    
    # Write cache file (using direct file I/O to handle absolute paths)
    os.makedirs(collection_folder, exist_ok=True)
    with open(cache_file_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=4)
    
    setattr(vdb, 'reranker', reranker)
    setattr(vdb, 'corpus_folder', corpus_folder)
    setattr(vdb, 'collection_folder', collection_folder)
    setattr(vdb, 'collection_name', collection_name)
    return vdb






if __name__ == "__main__":
    collection_name = "corpus1"
    vdb_type = "numpy"
    corpus_folder = os.path.abspath(fr"tests/data/{collection_name}")
    collection_folder = os.path.abspath(f"tests/data/vdb/{vdb_type}/{collection_name}")
    vdb = make_vectordb(collection_name, corpus_folder, collection_folder)

    context = vdb.retrieve_documents("I want to shift into another plane. What spell should I use?")
    print(context)
