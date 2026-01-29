import lib.ai.raptor
from lib.tools import findPath

def test_raptor_wilmott():
    raptor = lib.ai.raptor.create_raptor_ollama(
        corpus_folder=r"D:\rob\Wilmott Magazine",
        persist_dir="./storage/raptor/wilmott",
        collection_name="willmott",
        index_llm_model="qwen2.5-coder:latest",
        query_llm_model="gemma3:12b",
        embed_model="nomic-embed-text"
    )



if __name__ == "__main__":
    test_raptor_wilmott()