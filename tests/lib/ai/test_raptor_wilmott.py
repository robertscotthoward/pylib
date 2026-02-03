import pytest
from pathlib import Path
from lib.tools import findPath


def test_raptor_wilmott():
    from lib.ai.raptor import process_corpus
    from lib.ai.raptor import create_raptor_ollama
    corpus_folder=r"D:\rob\Wilmott Magazine"
    process_corpus(corpus_folder)
    raptor = create_raptor_ollama(
        corpus_folder=corpus_folder,
        persist_dir="./storage/raptor/wilmott",
        collection_name="wilmott",
        index_llm_model="qwen2.5-coder:latest",
        query_llm_model="gemma3:12b",
        embed_model="nomic-embed-text",
        timeout=600.0  # 10 minutes timeout for Ollama inference
    )

    answer = raptor.query("What do many sign errors stem from?")
    print(answer)


def test_ollama_index_wilmott():
    from lib.ai.orchestration import create_local_rag, process_corpus
    
    corpus_folder = findPath("D:\\rob\Wilmott Magazine")
    collection_name = "wilmott"
    persist_dir = Path(r"D:\rob\rag\vectordb\rob\chroma")
    persist_dir.mkdir(parents=True, exist_ok=True)

    process_corpus(corpus_folder)

    rag = create_local_rag(
        collection_name=collection_name,
        persist_dir=persist_dir,
        corpus_folder=corpus_folder,
    )
    answer = rag.query("What do many sign errors stem from?")
    print(answer)
    



# Skip all tests in this file
pytestmark = pytest.mark.skip(reason="Skip all tests in this file")

if __name__ == "__main__":
    # test_raptor_wilmott()
    test_ollama_index_wilmott()