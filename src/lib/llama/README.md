SentenceTransformerRerank - a Node Postprocessor that implements "Cross-Encoder" reranking. While standard vector search (Bi-Encoders) is fast, it can be "blunt." This class adds a second, more intelligent layer of filtering to your RAG pipeline. Two stages: bi-encoder and cross-encoder.
reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-2-v2", # A high-speed, standard choice
    top_n=3 # The final number of high-quality nodes sent to the LLM
)


VectorStoreIndex

StorageContext - where the VectorStoreIndex is kept.
* Ephemeral: chromadb.EphemeralClient() - In memory testing, CI/CD, and temporary scripts.
* Persistent: chromadb.PersistentClient(path=""..."") - Local RAG apps, desktop tools, and persistence.
* HTTP: chromadb.HttpClient(host=""..."", port=...) - Production apps with a dedicated database server.



pip install llama-index-postprocessor-sbert-rerank
pip install llama-index-embeddings-ollama

from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)



# 1. Create a persistent client pointing to your local folder
# If the folder doesn't exist, Chroma will create it for you.
db = chromadb.PersistentClient(path="./my_chroma_db")

# 2. Get or create your collection
chroma_collection = db.get_or_create_collection("my_rag_collection")

# 3. Assign it to the LlamaIndex VectorStore
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 4. Set up the StorageContext
storage_context = StorageContext.from_defaults(vector_store=vector_store)


query_engine = index.as_query_engine(
    similarity_top_k=5,             # Number of chunks to grab initially
    node_postprocessors=[reranker],  # Your SentenceTransformerRerank
    response_mode="compact",        # How the LLM bundles multiple chunks
    streaming=True                  # Stream the answer character-by-character
)

I have Ollama running locally. I want to use

From the above, create a RAG system with these parameters:
* corpus_folder = "C:\Rob\GitHub\robertscotthoward\python\pylib\tests\data\corpus1"
  * This folder contains text files that end with extension ".cleaned"
* ChromaVectorStore
* embedding_model_name="nomic-embed-text:latest",
* chromadb.PersistentClient with path = "D:\rob\rag\vectordb\test"
* reranker_model = "cross-encoder/ms-marco-MiniLM-L-2-v2"
* llm_model = "llama3.1:8b"


Incorporate this code if possible:

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

# 1. Initialize the Nomic model via Ollama
# Make sure you've run 'ollama pull nomic-embed-text' in your terminal first
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url="http://localhost:11434", # Default local Ollama address
    embed_batch_size=10
)

# 2. Set it globally so your VectorStoreIndex uses it automatically
Settings.embed_model = embed_model

query_engine = index.as_query_engine(
    similarity_top_k=5,             # Number of chunks to grab initially
    node_postprocessors=[reranker],  # Your SentenceTransformerRerank
    response_mode="compact",        # How the LLM bundles multiple chunks
    streaming=True                  # Stream the answer character-by-character
)