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


class RagBase:
    """Base class for RAG (Retrieval-Augmented Generation) systems."""

    def find_documents(self, question: str) -> List[Document]:
        """Return a list of documents that are relevant to the question."""
        raise NotImplementedError("Subclasses must implement this method.")

    def query(self, question: str) -> str:
        """Return a response to the question based on the relevant documents."""
        raise NotImplementedError("Subclasses must implement this method.")

    def index(self, documents: List[Document], rebuild: bool = False) -> None:
        """
        Index the documents into the vector store.
        
        Args:
            documents: List of documents to index
            rebuild: If True, rebuild the index from scratch; else add to existing index
        """
        raise NotImplementedError("Subclasses must implement this method.")


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
        if rebuild or self._index is None:
            # Create a new VectorStoreIndex from documents
            print(f"Creating VectorStoreIndex with {len(documents)} documents...")
            self._index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                embed_model=self.embed_model,
            )
            print("Index created successfully!")
        else:
            # Add documents to existing index
            print(f"Adding {len(documents)} documents to existing index...")
            for doc in documents:
                self._index.insert(doc)
            print("Documents added successfully!")


def test_llamaindex():
    """Test the LlamaIndex RAG implementation with local Ollama models."""
    import chromadb

    # Setup Chroma
    collection_name = "test_collection"
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Create RAG instance with Ollama models
    rag = LlamaIndexRAG(
        collection_name,
        vector_store,
        llm_model="gemma3:12b",
        embed_model_name="nomic-embed-text",
    )

    # Create sample documents
    sample_docs = [
        Document(text="The capital of France is Paris."),
        Document(text="The capital of Germany is Berlin."),
        Document(text="The capital of Italy is Rome."),
    ]

    # Index documents
    rag.index(sample_docs, rebuild=True)

    # Query
    response = rag.query("What is the capital of France?")
    print(f"Response: {response}")


if __name__ == "__main__":
    test_llamaindex()
