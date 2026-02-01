"""
Here are the interface classes and the factory functions for the orchestration of the AI pipeline.
"""
from typing import List
from lib.tools import *
from llama_index.core import Document




# ================================================================================
# CORPUS PROCESSING

# The extension of the cleaned files. When a document is read, it is converted to a text file, cleaned of all metadata and strange characters, and saved with this extension.
CLEANED_EXTENSION = '.cleaned'

class FilterTextFiles:
    def __init__(self):
        self.seen_hash = set()

    def should_keep_file(self, text):
        # 1. Normalize: Lowercase, remove special chars, collapse whitespace
        clean_text = re.sub(r'\W+', ' ', text).lower().strip()
        
        # 2. Filter by length (tokens are better than characters)
        if len(clean_text.split()) < 20: # Example: skip if less than 20 words
            return False

        # 3. Fuzzy match check (using a simple hash of the normalized text)
        text_hash = hash(clean_text)
        
        if text_hash not in self.seen_hash:
            self.seen_hash.add(text_hash)
            return True
        return False




def process_corpus(corpus_folder):
    """
    Ensure that all files in the corpus are converted to text, cleaned, and written to a sibling file with the extension CLEANED_EXTENSION.
    """
    import lib.ai.fileconvert
    filter = FilterTextFiles()
    def filter_func(text):
        return filter.should_keep_file(text)
    lib.ai.fileconvert.all_files_to_text(corpus_folder, CLEANED_EXTENSION, filter=filter_func)




# ================================================================================
# RAG BASE CLASS

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




def create_llamaindex_rag(
    collection_name: str,
    vector_store,
    llm_model: str = "gemma3:12b",
    embed_model_name: str = "nomic-embed-text",
    llm = None,
    embed_model = None,
) -> RagBase:
    """Create a LlamaIndex RAG instance."""
    from llama_index.llms.ollama import Ollama
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.embeddings.ollama import OllamaEmbedding
    from lib.ai.llamaindex import LlamaIndexRAG
    from llama_index.core.vector_stores.types import BasePydanticVectorStore

    assert_type(llm, Ollama)
    assert_type(vector_store, BasePydanticVectorStore)
    assert_type(llm_model, str)
    assert_type(embed_model_name, str)
    assert_type(embed_model, OllamaEmbedding)

    return LlamaIndexRAG(collection_name, vector_store, llm_model, embed_model_name, llm, embed_model)


def create_chroma_vector_store(
    collection_name: str,
    persist_dir: str,
):
    """Create a Chroma vector store.
    @collection_name: Name of the collection
    @persist_dir: Directory where Chroma data will be persisted
    """
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from pathlib import Path
    chroma_db_dir = Path(persist_dir) / collection_name
    chroma_db_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(chroma_db_dir))
    return ChromaVectorStore(chroma_collection=chroma_client.get_or_create_collection(collection_name))


def create_local_rag(
    collection_name: str,
    persist_dir: str,
    corpus_folder: str = None,
    vector_store = None,
    llm = None,
    embed_model = None,
    rebuild = False,
):
    """
    Create a RAG instance using local models using default configurations.
    @collection_name: The name of the collection.
    @corpus_folder: The folder containing the corpus. If None, the corpus will not be indexed.
    @persist_dir: The folder where the index will be saved.
    @vector_store: The vector store to use.
    @llm_model: The model to use for the LLM.
    @embed_model_name: The model to use for the embedding model.
    @llm: The LLM to use.
    @embed_model: The embedding model to use.
    @rebuild: If True, rebuild the index from scratch; else add to existing index. Defaults to False.
    """
    from llama_index.llms.ollama import Ollama
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.embeddings.ollama import OllamaEmbedding
    from lib.ai.llamaindex import LlamaIndexRAG
    from llama_index.core.vector_stores.types import BasePydanticVectorStore
    from llama_index.core import SimpleDirectoryReader

    if vector_store is None:
        vector_store = create_chroma_vector_store(collection_name, persist_dir)
    if llm is None:
        llm_model = "gemma3:12b"
        llm = Ollama(model=llm_model, request_timeout=300.0)
    if embed_model is None:
        embed_model_name = "nomic-embed-text"
        embed_model = OllamaEmbedding(model_name=embed_model_name, request_timeout=300.0)

    assert_type(llm, Ollama)
    assert_type(vector_store, BasePydanticVectorStore)
    assert_type(llm_model, str)
    assert_type(embed_model_name, str)
    assert_type(embed_model, OllamaEmbedding)

    rag = LlamaIndexRAG(collection_name, vector_store, llm_model, embed_model_name, llm, embed_model)
    
    # Index the corpus if provided
    if corpus_folder is not None:
        print(f"Loading documents from {corpus_folder}...")
        documents = SimpleDirectoryReader(
            corpus_folder,
            recursive=True,
            required_exts=[CLEANED_EXTENSION]
        ).load_data()
        print(f"Loaded {len(documents)} documents.")
        
        if rebuild:
            rag.index(documents, rebuild=True)
            print(f"Indexing complete!")
        else:
            # Load only modified/new documents
            from pathlib import Path
            import os
            
            # Get the modification time of the Chroma database
            chroma_db_dir = Path(persist_dir) / collection_name
            if chroma_db_dir.exists():
                db_mtime = os.path.getmtime(chroma_db_dir)
                print(f"Chroma database last modified: {db_mtime}")
                
                # First, load the existing index
                print("Loading existing index...")
                rag._load_existing_index()
                
                # Filter documents that were modified after the database
                modified_docs = []
                for doc in documents:
                    doc_path = Path(doc.metadata.get('file_path', ''))
                    if doc_path.exists():
                        doc_mtime = os.path.getmtime(doc_path)
                        if doc_mtime > db_mtime:
                            modified_docs.append(doc)
                
                print(f"Found {len(modified_docs)} modified documents out of {len(documents)} total.")
                if modified_docs:
                    print(f"Indexing {len(modified_docs)} modified documents...")
                    rag.index(modified_docs, rebuild=False)
                    print(f"Indexing complete!")
                else:
                    print("No modified documents to index.")
            else:
                # Database doesn't exist, index all documents
                print("Chroma database not found. Indexing all documents...")
                rag.index(documents, rebuild=True)
                print(f"Indexing complete!")
    
    return rag








def test():
    create_llamaindex_rag()

if __name__ == "__main__":
    test()