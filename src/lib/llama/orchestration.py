import os
import chromadb
from lib.ai.fileconvert import all_files_to_text
from lib.tools import *
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.ingestion import IngestionPipeline, IngestionCache



def create_rag_system(corpus_folder, reranker_model, llm_model, embedding_model_name, persist_dir, collection_name):
    """
    @corpus_folder is the folder containing the corpus to use.
    @reranker_model is the model to use for reranking.
    @llm_model is the model to use for the LLM.
    @embedding_model_name is the model to use for the embedding model.
    @persist_dir is the directory to use for the persistent storage.
    @collection_name is the name of the collection to use.
    """

    all_files_to_text(corpus_folder, cleaned_extension=".cleaned", overwrite=True)
    
    # --- STEP 1: Configure Local Models (via Ollama) ---
    # Use Nomic for embeddings and Llama 3 for the final answer
    Settings.embed_model = OllamaEmbedding(model_name=embedding_model_name)
    Settings.llm = Ollama(model=llm_model, request_timeout=120.0)

    # --- STEP 2: Setup Persistent Local Storage ---
    db = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # --- STEP 3: Create or Load the Index ---
    # Try to load existing index, otherwise create a new one
    try:
        # Load the existing index from persistent storage
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        print("Found existing index. Preparing to refresh...")
    except Exception as e:
        print(f"No existing index found ({type(e).__name__}). Creating a new one.")
        # Fallback to initial creation if index doesn't exist
        documents = SimpleDirectoryReader(
            input_dir=corpus_folder,
            recursive=True,
            required_exts=[".cleaned"]
        ).load_data()
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        print(f"Created new index with {len(documents)} documents.")
    
    # --- STEP 3b: Refresh Index with Current Files ---
    # Load CURRENT files from the folder
    new_documents = SimpleDirectoryReader(
        input_dir=corpus_folder,
        recursive=True,
        required_exts=[".cleaned"]
    ).load_data()
    
    # Refresh the index - only embeds/inserts what is NEW or CHANGED
    refreshed_docs = index.refresh_ref_docs(new_documents)
    print(f"Index refresh complete. Updated {sum(refreshed_docs)} documents.")

    # --- STEP 4: Initialize the Reranker ---
    # We pull a local cross-encoder to refine the results
    reranker = SentenceTransformerRerank(
        model=reranker_model, 
        top_n=3
    )

    # --- STEP 5: Create the Query Engine ---
    query_engine = index.as_query_engine(
        similarity_top_k=10,             # Initial vector search pulls 10 chunks
        node_postprocessors=[reranker],  # Reranker trims those 10 down to 3
        response_mode="compact"
    )
    return query_engine




def create_rag_system_with_defaults(corpus_folder, persist_dir, collection_name):
    """
    @corpus_folder is the folder containing the corpus to use.
    @persist_dir is the directory to use for the persistent storage.
    @collection_name is the name of the collection to use.
    """
    reranker_model = "cross-encoder/ms-marco-MiniLM-L-2-v2"
    llm_model = "llama3.1:8b"
    embedding_model_name = "nomic-embed-text:latest"
    return create_rag_system(corpus_folder, reranker_model, llm_model, embedding_model_name, persist_dir, collection_name)



def test1():
    corpus_folder = findPath("tests/data/corpus1")
    persist_dir = findPath("tests/data/rag")
    collection_name = "corpus1"
    query_engine = create_rag_system_with_defaults(corpus_folder, persist_dir, collection_name)

    prompt = readText(findPath("tests/data/prompts/prompt1.txt"))
    response = query_engine.query(
        f"What spell should I use to shift into another plane? Recommend a spell and explain why you chose it. Only use spells from the book.\n\n{prompt}"
        )
    print(f"\nResponse: {response}")




if __name__ == "__main__":
    test1()
