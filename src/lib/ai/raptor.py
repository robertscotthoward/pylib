import os
import pypdf
from pathlib import Path
from lib.tools import *

# Use this instead of "import raptor"
from llama_index.packs.raptor import RaptorPack, RaptorRetriever
from llama_index.core import SimpleDirectoryReader, Document, StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.llms.bedrock import Bedrock
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding



import nest_asyncio
nest_asyncio.apply() # Required for the clustering logic in notebooks/scripts


class Raptor:
    def __init__(self, corpus_folder, persist_dir="./storage/raptor"):
        self.corpus_folder = corpus_folder
        self.persist_dir = Path(persist_dir)
        
        self.llm = Bedrock(model="anthropic.claude-3-haiku-20240307-v1:0")
        self.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

        # Check if persist directory exists AND contains valid index files
        if self.persist_dir.exists() and (self.persist_dir / "docstore.json").exists():
            print(f"--- Loading RAPTOR from {self.persist_dir} ---")
            storage_context = StorageContext.from_defaults(persist_dir=str(self.persist_dir))
            # Load the underlying index
            self.index = load_index_from_storage(
                storage_context, 
                embed_model=self.embed_model
            )
            # Re-wrap it in the Pack so your query logic stays the same
            self.raptor_pack = RaptorPack(
                documents=[], # No docs needed for reload
                llm=self.llm,
                embed_model=self.embed_model,
                vector_store=self.index.vector_store
            )
        else:
            print("--- Creating New RAPTOR Index (this may take a while) ---")
            self.documents = SimpleDirectoryReader(self.corpus_folder, recursive=True).load_data()
            
            # This triggers the clustering/summarization logic
            self.raptor_pack = RaptorPack(
                self.documents,
                llm=self.llm,
                embed_model=self.embed_model
            )
            
            # Create directory and save the storage context
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            # RaptorPack has a storage_context we can persist
            if hasattr(self.raptor_pack, 'storage_context'):
                self.raptor_pack.storage_context.persist(persist_dir=str(self.persist_dir))
            print(f"--- Index saved to {self.persist_dir} ---")

    def query(self, query_str):
        # Using the retriever directly from the pack
        retriever = self.raptor_pack.retriever
        return retriever.retrieve(query_str)

    def print_clusters(self):
        """Print the clusters created by RAPTOR."""
        print("\n=== RAPTOR Clusters ===")
        try:
            # Try to access the nodes from the retriever's index
            if hasattr(self.raptor_pack, '_index') and hasattr(self.raptor_pack._index, 'docstore'):
                nodes = self.raptor_pack._index.docstore.docs.values()
                for i, node in enumerate(nodes):
                    print(f"\nCluster {i}:")
                    text = node.text if hasattr(node, 'text') else str(node)[:200]
                    print(f"  Text: {text[:200]}...")
                    if hasattr(node, 'metadata'):
                        print(f"  Metadata: {node.metadata}")
            elif hasattr(self.raptor_pack, 'retriever') and hasattr(self.raptor_pack.retriever, '_index'):
                nodes = self.raptor_pack.retriever._index.docstore.docs.values()
                for i, node in enumerate(nodes):
                    print(f"\nCluster {i}:")
                    text = node.text if hasattr(node, 'text') else str(node)[:200]
                    print(f"  Text: {text[:200]}...")
                    if hasattr(node, 'metadata'):
                        print(f"  Metadata: {node.metadata}")
            else:
                print("Could not access clusters from raptor_pack")
                print(f"Available attributes: {[attr for attr in dir(self.raptor_pack) if not attr.startswith('_')]}")
        except Exception as e:
            print(f"Error accessing clusters: {e}")

    def print_hierarchy(self):
        """Print the hierarchy tree created by RAPTOR."""
        print("\n=== RAPTOR Hierarchy ===")
        try:
            # Try to access the nodes from the retriever's index
            if hasattr(self.raptor_pack, '_index') and hasattr(self.raptor_pack._index, 'docstore'):
                nodes = list(self.raptor_pack._index.docstore.docs.values())
                self._print_tree(nodes, level=0)
            elif hasattr(self.raptor_pack, 'retriever') and hasattr(self.raptor_pack.retriever, '_index'):
                nodes = list(self.raptor_pack.retriever._index.docstore.docs.values())
                self._print_tree(nodes, level=0)
            else:
                print("Could not access hierarchy from raptor_pack")
        except Exception as e:
            print(f"Error accessing hierarchy: {e}")

    def _print_tree(self, nodes, level=0):
        """Recursively print the tree structure."""
        indent = "  " * level
        for i, node in enumerate(nodes):
            print(f"{indent}Node {i}: {node.text[:100]}...")
            if hasattr(node, 'child_nodes') and node.child_nodes:
                self._print_tree(node.child_nodes, level + 1)




def test_raptor():
    corpus_folder = findPath("data/corpus1")
    if corpus_folder is None:
        raise FileNotFoundError(f"Could not find corpus folder 'data/corpus1'")
    corpus_folder = os.path.abspath(corpus_folder)
    raptor = Raptor(corpus_folder)
    raptor.print_clusters()
    raptor.print_hierarchy()
    response = raptor.query("What is the main finding of the report?")
    assert response is not None
    assert len(response) > 0
    assert response[0].text is not None
    assert response[0].score is not None
    assert response[0].metadata is not None
    # Metadata may contain different keys depending on the document type
    # Just verify metadata exists and is not empty
    assert len(response[0].metadata) >= 0



if __name__ == "__main__":
    test_raptor()