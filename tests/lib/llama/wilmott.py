import glob

from llama_index.llms.ollama import Ollama
from requests import ReadTimeout
from lib.llama.orchestration import create_rag_system, create_rag_system_with_defaults
from lib.tools import *



corpus_folder = findPath(r"D:\rob\Wilmott Magazine")
persist_dir = r"D:\rob\rag\vectordb\rob\chroma\wilmott"
collection_name = "wilmott"
ensureFolder(persist_dir)




def getRagEngine():
    reranker_model = "cross-encoder/ms-marco-MiniLM-L-2-v2"
    embedding_model_name = "nomic-embed-text:latest"
    llm_model = "mistral-nemo:12b" # For summarizing large documents 128K tokens
    llm_model = "hf.co/lmstudio-community/Qwen2.5-7B-Instruct-1M-GGUF:Q4_K_M" # For summarizing large documents 1M tokens. Custom made.
    llm_model = "qwen2.5-coder:32b"
    llm_model = "mistral-nemo-128k" # For summarizing large documents 128K tokens. Custom made.
    llm_model = "mistral:latest"
    llm_model = "llama3.1:8b"
    desired_context_size = 4096  # context window
    similarity_top_k = 20
    reranker_top_n = 5
    max_tokens = 1024 # Output tokens. Good for a lengthy summary.
    query_engine = create_rag_system(
        corpus_folder=corpus_folder,
        reranker_model=reranker_model,
        llm_model=llm_model,
        actual_context_window=desired_context_size,
        embedding_model_name=embedding_model_name,
        persist_dir=persist_dir,
        collection_name=collection_name,
        similarity_top_k=similarity_top_k,
        reranker_top_n=reranker_top_n,
        max_tokens=max_tokens)
    return query_engine



def getQueryEngine() -> Ollama:
    llm_model = "mistral-nemo:12b"
    max_tokens = 8192
    actual_context_window = 128000
    llm = Ollama(
        model=llm_model, 
        request_timeout=600.0,  # 10 minutes - should be enough for 20 second responses
        max_tokens=max_tokens,
        context_window=actual_context_window,
        additional_kwargs={
            "num_ctx": actual_context_window,  # Tell Ollama to use full context window
            "num_predict": max_tokens,  # Explicitly set max output tokens
        }
    )
    return llm



def queryRag():
    query_engine = getRagEngine()
    prompt = readText(findPath("tests/data/prompts/prompt1.txt"))
    response = query_engine.query(
        f"Summarize the general theme of the magazine.\n\n{prompt}"
        )
    print(f"\nResponse: {response}")


def SummarizeMagazines():
    # Get all distinct magazine issues as YYYYMM
    matcher = re.compile(r'wilmott-(?P<Issue>\d{4}\d{2})')
    llm = getQueryEngine()
    issues = {}
    for file in glob.glob(os.path.join(corpus_folder, '*.cleaned')):
        match = matcher.search(file)
        if match:
            magazine_issue = match.group('Issue')
            issue = issues.get(magazine_issue, {
                "files": [],
                "summary": "",
            })
            issue["files"].append(file)
            issues[magazine_issue] = issue

    context_size = llm._model_kwargs["num_ctx"]

    with Spy("Summarize") as spy:
        sumAll = ""
        prompt = readText(findPath("tests/data/prompts/prompt2.txt"))
        for key, issue in issues.items():
            sumIssue = ""
            date = f"{key[:4]}-{key[4:]}"
            for file in issue["files"]:
                print(f"Summarizing {key}: {file}")
                fileSumPath = file.replace('.cleaned', '.summary.txt')
                if os.path.exists(fileSumPath):
                    continue
                text = readText(file)
                p = prompt.replace("{ISSUE}", date) + text
                # Calculate the number of tokens in the prompt and response
                p_tokens = len(p) / 4
                if p_tokens > context_size:
                    print(f"Prompt too long: {p_tokens} tokens")
                    continue
                try:
                    # Use streaming to get response incrementally
                    response_text = ""
                    print(f"  Generating summary...", end="", flush=True)
                    for chunk in llm.stream_complete(p):
                        response_text += chunk.delta
                        print(".", end="", flush=True)
                    print(" Done!")
                    writeText(fileSumPath, response_text)
                    sumIssue += f"WILMOTT ISSUE {date}: {response_text}\n\n\n\n"
                except ReadTimeout as e:
                    print(f"ReadTimeout: {e}")
                    pass
                except Exception as e:
                    print(f"Error: {e}")
                    pass
            issue["summary"] = sumIssue
            writeYaml(corpus_folder + f"{key}.yaml", issue)
            sumAll += f"WILMOTT ISSUE {date}\n{sumIssue}\n\n\n\n"
        writeText(corpus_folder + f"{key}.txt", sumAll)


if __name__ == "__main__":
    #queryRag()
    SummarizeMagazines()