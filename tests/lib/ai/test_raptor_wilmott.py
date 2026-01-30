import lib.ai.raptor
import lib.ai.fileconvert
from lib.tools import findPath

def test_raptor_wilmott():
    corpus_folder=r"D:\rob\Wilmott Magazine"
    lib.ai.raptor.process_corpus(corpus_folder)
    raptor = lib.ai.raptor.create_raptor_ollama(
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



if __name__ == "__main__":
    test_raptor_wilmott()