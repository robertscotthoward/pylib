"""
LlamaIndex RAG implementation using VectorStoreIndex and Chroma vector store.

This module provides a base class for RAG (Retrieval-Augmented Generation) systems
and a concrete implementation using LlamaIndex's VectorStoreIndex with local Ollama models.
"""

from typing import List, Optional
from llama_index.core import (
    StorageContext,
    Document,
    VectorStoreIndex,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from lib.ai.orchestration import *




class LlamaIndexRAG(RagBase):
    """RAG implementation using LlamaIndex VectorStoreIndex with local Ollama models."""

    def __init__(
        self,
        collection_name: str,
        vector_store: ChromaVectorStore,
        llm_model: str = "gemma3:12b",
        embed_model_name: str = "nomic-embed-text",
        llm: Optional[Ollama] = None,
        embed_model: Optional[OllamaEmbedding] = None,
    ):
        """
        Initialize the LlamaIndex RAG system with Ollama models.
        @collection_name: Name of the Chroma collection
        @vector_store: ChromaVectorStore instance
        @llm_model: Ollama model name for LLM (default: gemma3:12b)
        @embed_model_name: Ollama model name for embeddings (default: nomic-embed-text)
        @llm: Custom Ollama LLM instance (optional, uses llm_model if not provided)
        @embed_model: Custom OllamaEmbedding instance (optional, uses embed_model_name if not provided)
        """
        self.collection_name = collection_name
        self.vector_store = vector_store
        
        # Use provided models or create default Ollama instances
        self.llm = llm or Ollama(model=llm_model, request_timeout=300.0)
        self.embed_model = embed_model or OllamaEmbedding(
            model_name=embed_model_name, 
            request_timeout=300.0
        )
        
        self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self._index = None


    def find_documents(self, question: str) -> List[Document]:
        """
        Find documents relevant to the question.
        @question: The question to search for
        Returns: List of relevant documents
        """
        if self._index is None:
            raise ValueError("Index not initialized. Call index_documents() first.")

        retriever = self._index.as_retriever(similarity_top_k=5)
        results = retriever.retrieve(question)
        return [result.node for result in results]


    def query(self, question: str) -> str:
        """
        Query the index and return a response.
        @question: The question to answer
        Returns: The answer string.
        """
        if self._index is None:
            raise ValueError("Index not initialized. Call index_documents() first.")

        # Use the retriever to find relevant documents
        retriever = self._index.as_retriever(similarity_top_k=5)
        results = retriever.retrieve(question)
        
        # Format the results into a response
        if not results:
            return "No relevant documents found."
        
        # Combine the retrieved documents
        context = "\n\n".join([result.node.text for result in results])
        
        # Use the LLM to generate a response based on the context
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        response = self.llm.complete(prompt)
        return str(response)


    def index(self, documents: List[Document], rebuild: bool = False) -> None:
        """
        Index documents into the vector store.
        @documents: List of documents to index
        @rebuild: If True, rebuild from scratch; else add to existing
        """
        try:
            if rebuild or self._index is None:
                # Create a new VectorStoreIndex from documents
                print(f"Creating VectorStoreIndex with {len(documents)} documents...")
                self._index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=self.storage_context,
                    embed_model=self.embed_model,
                    show_progress=True,
                )
                print("Index created successfully!")
            else:
                # Add documents to existing index
                print(f"Adding {len(documents)} documents to existing index...")
                for i, doc in enumerate(documents, 1):
                    file_path = doc.metadata.get('file_path', 'unknown')
                    print(f"  ✓ [{i:3d}/{len(documents)}] {file_path}")
                    self._index.insert(doc)
                print("Documents added successfully!")
        except Exception as e:
            print(f"Error during indexing: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _load_existing_index(self) -> None:
        """
        Load an existing index from the vector store.
        @returns: None (sets self._index)
        """
        try:
            if self._index is None:
                print("Loading existing VectorStoreIndex from storage...")
                self._index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store,
                    embed_model=self.embed_model,
                )
                print("Index loaded successfully!")
        except Exception as e:
            print(f"Error loading existing index: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise




# ============================== TESTS ==============================

def test_llamaindex():
    pass

def test_with_chroma_store():
    """Test the LlamaIndex RAG implementation with local Ollama models."""
    import chromadb
    from pathlib import Path

    # Setup Chroma with persistence
    collection_name = "test_collection"
    persist_dir = Path("./storage/chroma/llamaindex_test")
    persist_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created persist directory: {persist_dir.absolute()}")
    
    try:
        chroma_client = chromadb.PersistentClient(path=str(persist_dir))
        print(f"Chroma client created successfully")
        
        chroma_collection = chroma_client.get_or_create_collection(collection_name)
        print(f"Chroma collection '{collection_name}' created/loaded")
        
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        print(f"ChromaVectorStore initialized")

        # Create RAG instance with Ollama models
        rag = LlamaIndexRAG(
            collection_name,
            vector_store,
            llm_model="gemma3:12b",
            embed_model_name="nomic-embed-text",
        )
        print(f"LlamaIndexRAG instance created")

        # Create sample documents
        sample_docs = [
            Document(text="The capital of France is Paris."),
            Document(text="The capital of Germany is Berlin."),
            Document(text="The capital of Italy is Rome."),
        ]
        print(f"Created {len(sample_docs)} sample documents")

        # Index documents
        print("Indexing documents...")
        rag.index(sample_docs, rebuild=True)

        # Query
        print("Querying index...")
        response = rag.query("What is the capital of France?")
        print(f"Response: {response}")
        print(f"\nDatabase persisted to: {persist_dir.absolute()}")
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_with_chroma_store()
    test_llamaindex()
