from lib.llama.orchestration import create_rag_system_with_defaults
from lib.tools import *



corpus_folder = findPath(r"D:\rob\Wilmott Magazine")
persist_dir = r"D:\rob\rag\vectordb\rob\chroma\wilmott"
ensureFolder(persist_dir)
collection_name = "wilmott"
query_engine = create_rag_system_with_defaults(corpus_folder, persist_dir, collection_name)

prompt = readText(findPath("tests/data/prompts/prompt1.txt"))
response = query_engine.query(
    f"Summarize the general theme of the magazine.\n\n{prompt}"
    )
print(f"\nResponse: {response}")
