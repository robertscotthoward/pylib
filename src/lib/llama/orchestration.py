import glob
import os
from pathlib import Path
import chromadb
from lib.ai.fileconvert import all_files_to_text
from lib.tools import *
from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import SentenceTransformerRerank



def create_rag_system(
    corpus_folder, 
    reranker_model, 
    llm_model, 
    actual_context_window=None,
    embedding_model_name=None,   
    persist_dir=None, 
    collection_name=None, 
    similarity_top_k=20, 
    reranker_top_n=5, 
    max_tokens=512):
    """
    @corpus_folder is the folder containing the corpus to use.
    @reranker_model is the model to use for reranking.
    @llm_model is the model to use for the LLM.
    @actual_context_window is the actual context size for the LLM.
    @embedding_model_name is the model to use for the embedding model.
    @persist_dir is the directory to use for the persistent storage.
    @collection_name is the name of the collection to use.
    @similarity_top_k is the number of chunks to retrieve initially (default: 20).
    @reranker_top_n is the number of chunks to keep after reranking (default: 5).
    @max_tokens is the maximum number of tokens the LLM can output (default: 512).
    """


    rag_metadata_path = Path(persist_dir) / "rag_metadata.json"
    if os.path.exists(rag_metadata_path):
        rag_metadata = json.load(open(rag_metadata_path))
        last_updated = rag_metadata.get("last_updated", 0)
    else:
        rag_metadata = {}
        last_updated = 0


    all_files_to_text(corpus_folder, cleaned_extension=".cleaned", overwrite=True)
    
    if not actual_context_window:
        # Map of known model context windows
        model_context_windows = {
            "mistral-nemo:12b": 128000,
            "mistral-nemo-128k": 128000,
            "qwen2.5-coder:32b": 128000,  # Qwen2.5-Coder supports 128K tokens
            "qwen2.5-coder:latest": 128000,
            "llama3.1:8b": 8192,
            "gemma3:12b": 8192,
        }
        actual_context_window = model_context_windows.get(llm_model, 8192)

    # CONFIGURE LOCAL MODELS (via Ollama)
    # Use Nomic for embeddings and Qwen for the final answer
    Settings.embed_model = OllamaEmbedding(model_name=embedding_model_name)
    llm = Ollama(
        model=llm_model, 
        request_timeout=300.0,  # Increased timeout for larger models
        max_tokens=max_tokens,
        context_window=actual_context_window,
        additional_kwargs={
            "num_ctx": actual_context_window,  # Tell Ollama to use full context window
            "num_predict": max_tokens,  # Explicitly set max output tokens
        }
    )
    Settings.llm = llm
    Settings.context_window = actual_context_window

    context_size = actual_context_window or Settings.llm.metadata.context_window
    print(f"Model '{llm_model}' has a context window of {context_size} tokens.")

    # --- STEP 2: Setup Persistent Local Storage ---
    db = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)


    # CREATE OR LOAD THE INDEX
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
    

    # REFRESH INDEX WITH CURRENT FILES
    # For all *.cleaned files in corpus_folder, get the modification time.
    # If the modification time is greater than last_updated, then add the file to the index.
    new_documents = []
    total_documents = 0
    max_updated = last_updated + 1
    for file in glob.glob(os.path.join(corpus_folder, '*.cleaned')):
        total_documents += 1
        file_updated = os.path.getmtime(file)
        if file_updated > last_updated:
            max_updated = file_updated
            text = readText(file)
            new_documents.append(Document(text=text, metadata={'file_path': file}))
    print(f"Found {len(new_documents)} new files of {total_documents} in corpus_folder, max updated time: {max_updated}")
    

    # INSERT THE CHANGED DOCUMENTS DIRECTLY
    if new_documents:
        print(f"Inserting {len(new_documents)} updated documents...")
        for doc in new_documents:
            print(f"Embedding document: {doc.metadata['file_path']}")
            index.insert(doc)
        print(f"Index refresh complete. Updated {len(new_documents)} documents.")
        
        # Save the current timestamp to metadata file
        rag_metadata['last_updated'] = max_updated
        writeJson(rag_metadata_path, rag_metadata)
        print(f"Saved index metadata to {rag_metadata_path}")
    else: 
        print("No updated documents to refresh.")


    # INITIALIZE THE RERANKER
    # We pull a local cross-encoder to refine the results
    reranker = SentenceTransformerRerank(
        model=reranker_model, 
        top_n=reranker_top_n
    )

    # CREATE THE QUERY ENGINE
    query_engine = index.as_query_engine(
        llm=llm,  # Pass the LLM with max_tokens configured
        similarity_top_k=similarity_top_k, # Initial vector search pulls similarity_top_k chunks
        node_postprocessors=[reranker],  # Reranker trims those similarity_top_k matches down to reranker_top_n
        response_mode="compact"
    )

    rag_spec = {
        "last_updated": max_updated,
        "total_documents": total_documents,
        "new_documents": len(new_documents),
        "max_updated": max_updated,
        "similarity_top_k": similarity_top_k,
        "reranker_top_n": reranker_top_n,
        "rag": {
            "reranker_model": reranker_model,
            "llm_model": llm_model,
            "context_size": context_size,
            "embedding_model_name": embedding_model_name,
            "collection_name": collection_name,
            "vector_store": vector_store.__class__.__name__,
            "storage_context": storage_context.__class__.__name__,
            "index": index.__class__.__name__,
            "query_engine": query_engine.__class__.__name__,
            "reranker": reranker.__class__.__name__,
        }
    }

    writeJson(rag_metadata_path, rag_spec)
    print(f"Saved index specification to {rag_metadata_path}")

    setattr(query_engine, "max_tokens", max_tokens)
    setattr(query_engine, "context_size", context_size)
    setattr(query_engine, "reranker_model", reranker_model)
    setattr(query_engine, "llm_model", llm_model)
    setattr(query_engine, "embedding_model_name", embedding_model_name)
    setattr(query_engine, "persist_dir", persist_dir)
    setattr(query_engine, "collection_name", collection_name)
    setattr(query_engine, "similarity_top_k", similarity_top_k)
    setattr(query_engine, "reranker_top_n", reranker_top_n)
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



def testGetEngine():
    with Spy("Load") as spy:
        corpus_folder = findPath("tests/data/corpus1")
        persist_dir = findPath("tests/data/rag")
        collection_name = "corpus1"
        query_engine = create_rag_system_with_defaults(corpus_folder, persist_dir, collection_name)
        return query_engine




def test1():
    query_engine = testGetEngine()
    prompt = readText(findPath("tests/data/prompts/prompt1.txt"))
    
    with Spy("Query") as spy:
        response = query_engine.query(
            f"What spell should I use to shift into another plane? Recommend a spell and explain why you chose it. Only use spells from the book.\n\n{prompt}"
            )
        print(f"\nResponse: {response}")




def test2():
    query_engine = testGetEngine()
    prompt = readText(findPath("tests/data/prompts/prompt1.txt"))
    
    with Spy("Query") as spy:
        response = query_engine.query(
            f"What spell should I use to shift into another plane? Recommend a spell and explain why you chose it. Only use spells from the book.\n\n{prompt}"
            )
        print(f"\nResponse: {response}")




if __name__ == "__main__":
    test1()
