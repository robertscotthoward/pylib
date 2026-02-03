import chromadb
from lib.tools import findPath
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import SentenceTransformerRerank



def create_rag_system(corpus_folder, reranker_model, llm_model, embedding_model_name, persist_dir, collection_name):
    """
    @corpus_folder is the folder containing the corpus to use.
    @reranker_model is the model to use for reranking.
    @llm_model is the model to use for the LLM.
    @embedding_model_name is the model to use for the embedding model.
    @persist_dir is the directory to use for the persistent storage.
    @collection_name is the name of the collection to use.
    """
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
    # Load documents from a local folder called 'data'
    documents = SimpleDirectoryReader(corpus_folder).load_data()
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )

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
    response = query_engine.query("What spell should I use to shift into another plane? Recommend a spell and explain why you chose it. Only use spells from the book.")
    print(f"\nResponse: {response}")




if __name__ == "__main__":
    test1()
