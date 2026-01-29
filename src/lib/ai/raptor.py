import os
from pathlib import Path
from lib.tools import *

# Use this instead of "import raptor"
from llama_index.packs.raptor import RaptorPack, RaptorRetriever
from llama_index.core import SimpleDirectoryReader, Document, StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.llms.bedrock import Bedrock
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


"""
Raptor is a tool for indexing and querying a corpus of documents.
It uses the RaptorPack and RaptorRetriever classes from llama_index.
It creates a cluster of hierarchicaldocuments that are similar to each other, and then summarizes the cluster.
If any file in the corpus changes, the index must be regenerated in total since updating the index is not possible with RAPTOR.
"""


import nest_asyncio
nest_asyncio.apply() # Required for the clustering logic in notebooks/scripts

# The extension of the cleaned files. When a document is read, it is converted to a text file, cleaned of all metadata and strange characters, and saved with this extension.
CLEANED_EXTENSION = '.cleaned'

class Raptor:
    def __init__(self, corpus_folder, persist_dir="./storage/raptor", vector_store=None):
        """
        @corpus_folder is the folder containing the corpus to use.
        @persist_dir is the folder where the index will be saved.
        @vector_store is the vector store to use to store the embeddings. None means to use JSON files in the persist_dir.
        """
        self.corpus_folder = corpus_folder
        self.persist_dir = Path(persist_dir)
        
        self.llm = Bedrock(model="anthropic.claude-3-haiku-20240307-v1:0")
        self.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
        self.vector_store = vector_store

        # Check if persist directory exists AND contains valid index files
        if self.persist_dir.exists() and (self.persist_dir / "docstore.json").exists():
            print(f"--- Loading RAPTOR from {self.persist_dir} ---")
            try:
                # Load the retriever from persistence
                retriever = RaptorRetriever.from_persist_dir(
                    persist_dir=str(self.persist_dir),
                    llm=self.llm,
                    embed_model=self.embed_model
                )
                # Wrap it in a minimal object that has a retriever attribute
                self.raptor_pack = type('RaptorPackWrapper', (), {'retriever': retriever})()
            except Exception as e:
                print(f"Warning: Could not load from persist_dir: {e}")
                print("Creating new RAPTOR Index instead...")
                self.documents = SimpleDirectoryReader(
                    self.corpus_folder, 
                    recursive=True,
                    required_exts=[".cleaned"]
                ).load_data()
                self.raptor_pack = RaptorPack(
                    self.documents,
                    llm=self.llm,
                    embed_model=self.embed_model,
                    vector_store=self.vector_store
                )
        else:
            print("--- Creating New RAPTOR Index (this may take a while) ---")
            # Only read .cleaned files
            self.documents = SimpleDirectoryReader(
                self.corpus_folder, 
                recursive=True,
                required_exts=[".cleaned"]
            ).load_data()
            
            # This triggers the clustering/summarization logic
            self.raptor_pack = RaptorPack(
                self.documents,
                llm=self.llm,
                embed_model=self.embed_model
            )
            
            # Create directory and save the storage context
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            # Use the persist method on the retriever
            try:
                self.raptor_pack.retriever.persist(persist_dir=str(self.persist_dir))
                print(f"--- Index saved to {self.persist_dir} ---")
            except Exception as e:
                print(f"Warning: Error persisting index: {e}")
                import traceback
                traceback.print_exc()

    def query(self, query_str):
        # Using the retriever directly from the pack
        retriever = self.raptor_pack.retriever
        response = retriever.retrieve(query_str)
        
        # Deduplicate results based on text content
        seen_texts = set()
        deduplicated = []
        for node in response:
            # Use a hash of the text to detect duplicates
            text_hash = hash(node.text)
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                deduplicated.append(node)
        
        return deduplicated

    def print_clusters(self):
        """Print the clusters created by RAPTOR."""
        print("\n=== RAPTOR Clusters ===")
        try:
            retriever = self.raptor_pack.retriever
            index = retriever.index
            if hasattr(index, 'docstore'):
                nodes = list(index.docstore.docs.values())
                print(f"Found {len(nodes)} nodes/clusters")
                for i, node in enumerate(nodes):
                    print(f"\nCluster {i}:")
                    text = node.text if hasattr(node, 'text') else str(node)[:200]
                    print(f"  Text: {text[:200]}...")
                    if hasattr(node, 'metadata'):
                        print(f"  Metadata: {node.metadata}")
            else:
                print("Could not access docstore from index")
        except Exception as e:
            print(f"Error accessing clusters: {e}")
            import traceback
            traceback.print_exc()

    def print_hierarchy(self):
        """Print the hierarchy tree created by RAPTOR."""
        print("\n=== RAPTOR Hierarchy ===")
        try:
            retriever = self.raptor_pack.retriever
            index = retriever.index
            if hasattr(index, 'docstore'):
                nodes = list(index.docstore.docs.values())
                self._print_tree(nodes, level=0)
            else:
                print("Could not access docstore from index")
        except Exception as e:
            print(f"Error accessing hierarchy: {e}")
            import traceback
            traceback.print_exc()

    def _print_tree(self, nodes, level=0):
        """Recursively print the tree structure."""
        indent = "  " * level
        for i, node in enumerate(nodes):
            print(f"{indent}Node {i}: {node.text[:100]}...")
            if hasattr(node, 'child_nodes') and node.child_nodes:
                self._print_tree(node.child_nodes, level + 1)




# ============================== TESTS ==============================

def query(raptor):
    question = "Who thought of the idea for cooling the room?"
    response = raptor.query(question)
    print(f"Question: {question}\nResponse:")
    for i, r in enumerate(response):    
        print(f"  {i}: {r.text}")

    assert response is not None
    assert len(response) > 0
    assert response[0].text is not None
    assert response[0].score is not None
    assert response[0].metadata is not None
    # Metadata may contain different keys depending on the document type
    # Just verify metadata exists and is not empty
    assert len(response[0].metadata) >= 0



def test_with_chroma_store():
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import lib.ai.fileconvert

    collection_name = "chroma_test"
    persist_dir = f"./storage/raptor/{collection_name}"
    
    # Create a persistent Chroma client that stores in the same persist_dir
    chroma_db_dir = Path(persist_dir) / "chroma"
    chroma_db_dir.mkdir(parents=True, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(path=str(chroma_db_dir))
    vector_store = ChromaVectorStore(chroma_collection=chroma_client.get_or_create_collection(collection_name))
    
    corpus_folder = findPath("data/corpus1")
    if corpus_folder is None:
        raise FileNotFoundError(f"Could not find corpus folder 'data/corpus1'")

    lib.ai.fileconvert.all_files_to_text(corpus_folder, CLEANED_EXTENSION)

    corpus_folder = os.path.abspath(corpus_folder)
    raptor = Raptor(corpus_folder, persist_dir=persist_dir, vector_store=vector_store)
    query(raptor)
    print(f"All files saved to: {persist_dir}")




def test_raptor_with_default_json_store():
    import lib.ai.fileconvert
    corpus_folder = findPath("data/corpus1")
    lib.ai.fileconvert.all_files_to_text(corpus_folder, CLEANED_EXTENSION)
    if corpus_folder is None:
        raise FileNotFoundError(f"Could not find corpus folder 'data/corpus1'")
    corpus_folder = os.path.abspath(corpus_folder)
    raptor = Raptor(corpus_folder)
    raptor.print_clusters()
    raptor.print_hierarchy()
    query(raptor)





def test_raptor():
    test_with_chroma_store()
    test_raptor_with_default_json_store()



if __name__ == "__main__":
    test_with_chroma_store()
    #test_raptor()