from pathlib import Path
from lib.tools import findPath


def test_raptor_axiosfounder():
    from lib.ai.raptor import process_corpus
    from lib.ai.raptor import create_raptor_ollama
    corpus_folder=r"C:\Rob\GitHub\robertscotthoward\youtube-tools\cache\channels\axiomfounder"
    process_corpus(corpus_folder)
    raptor = create_raptor_ollama(
        corpus_folder=corpus_folder,
        persist_dir="./storage/raptor/axiosfounder",
        collection_name="axiosfounder",
        index_llm_model="qwen2.5-coder:latest",
        query_llm_model="gemma3:12b",
        embed_model="nomic-embed-text",
        timeout=600.0  # 10 minutes timeout for Ollama inference
    )

    answer = raptor.query("Create a checklist for building a startup.")
    print(answer)


def make_axiosfounder_rag():
    from lib.ai.orchestration import create_local_rag, process_corpus
    
    corpus_folder=r"C:\Rob\GitHub\robertscotthoward\youtube-tools\cache\channels\axiomfounder"
    collection_name = "axiomfounder"
    persist_dir = Path(r"D:\rob\rag\vectordb\rob\chroma")
    persist_dir.mkdir(parents=True, exist_ok=True)

    process_corpus(corpus_folder)

    rag = create_local_rag(
        collection_name=collection_name,
        persist_dir=persist_dir,
        corpus_folder=corpus_folder,
    )
    return rag


def test_ollama_index_axiosfounder():
    rag = make_axiosfounder_rag()
    answer = rag.query("Create a checklist for building a startup.")
    print(answer)
    




if __name__ == "__main__":
    test_ollama_index_axiosfounder()