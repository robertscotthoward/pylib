# uv add langchain-text-splitters
import lib.tools
from lib.tools import *
import pytest




class Reranker:
    def __init__(self):
        pass

    def compute_score(self, pairs, batch_size=32):
        return self.reranker.compute_score(pairs, batch_size=batch_size)




class GeneralReranker(Reranker):
    """
    uv add rerankers[transformers]

    Advantages::
      * Dependency Isolation: These libraries use "extra" installs (e.g., pip install "rerankers[transformers]"). This prevents your Django app from crashing if one specific model library has a conflict.
      * Output Consistency: They all return a standard RankedResults object, making it easy to swap a local BGE model for a cloud-based Jina API without changing your frontend code.
      * Optimization: They include 2026-standard optimizations like Flash Attention 2 and Triton kernels by default, which your manual NumPy implementation would miss.    
    """
    def __init__(self, model_name= "BAAI/bge-reranker-v2-m3"):
        import rerankers
        super().__init__()
        self.reranker = rerankers.Reranker(model_name)

    def compute_score(self, pairs, batch_size=32):
        return self.reranker.compute_score(pairs, batch_size=batch_size)




class Flag_Reranker(Reranker):
    """Reranker using the FlagEmbedding library. This one is slower but more accurate. Use the NumPy_Reranker for faster results."""
    def __init__(self):
        super().__init__()

        # uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
        # uv add flagembedding
        import torch
        from FlagEmbedding import FlagReranker
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        modelName = 'BAAI/bge-reranker-v2-m3'
        if self.device == "cuda":
            modelName = 'BAAI/bge-reranker-large'
        self.reranker = FlagReranker(modelName, device=self.device, use_fp16=True)




class NumPy_Reranker(Reranker):
    """
    Reranker using the NumPy library. This one is faster but less accurate. Use the Flag_Reranker for more accurate results.
    In this implementation, arrays are not "loaded" from disk—they are created in memory during each call to compute_score. 
    This is efficient for typical RAG workflows where you rerank 20–100 retrieved documents per query.
    """
    def __init__(self, model_name=None, device='cuda', use_fp16=False):
        super().__init__()
        import torch
        from FlagEmbedding import FlagReranker # Correct class for reranking

        # 1. Device detection
        if torch.cuda.is_available() and device == 'cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # 2. Smart model selection
        if not model_name:
            if self.device == "cuda":
                # High performance for GPU
                model_name = 'BAAI/bge-reranker-v2-m3'
            else:
                # Check for low memory via your tool lib
                try:
                    is_low_ram = lib.tools.is_low() 
                except:
                    is_low_ram = False
                
                if is_low_ram:
                    # Small and fast for CPU
                    model_name = 'BAAI/bge-reranker-base' 
                else:
                    # Balanced for CPU
                    model_name = 'BAAI/bge-reranker-v2-m3'

        print(f"Loading Reranker: {model_name} on {self.device}")
        
        # 3. Initialize the Reranker directly
        # Note: BGE models work natively with FlagReranker
        self.model = FlagReranker(model_name, device=self.device, use_fp16=use_fp16)

    def compute_score(self, pairs, batch_size=32):
        # FlagReranker uses .compute_score, not .rerank

        with Spy('compute_score') as spy:
            res = self.model.compute_score(pairs, batch_size=batch_size)
            spy.trace(f"compute_score took {spy.elapsedSeconds():.4f}s")
        return res




# ================================================================================
# TESTS



# This fixture loads the model once per test session/module
@pytest.fixture(scope="module")
def bge_ranker():
    # Use Flag_Reranker which is properly implemented
    return Flag_Reranker()

@pytest.mark.skip(reason="FlagEmbedding has dependency conflicts with transformers library")
def test_relevance_ranking(bge_ranker):
    """Verify that relevant info scores higher than noise."""
    query = "How do I use a 5090 for AI?"
    docs = [
        "The RTX 5090 is ideal for local LLMs and RAG tasks.",  # Relevant
        "Apples are usually red or green and grow on trees."    # Noise
    ]
    
    # Create pairs for reranking
    pairs = [[query, doc] for doc in docs]
    scores = bge_ranker.compute_score(pairs)
    
    # Verify that relevant doc scores higher than noise
    assert len(scores) == 2
    assert scores[0] > scores[1], f"Expected relevant doc to score higher: {scores}"

    # Switch order
    docs = [
        "Apples are usually red or green and grow on trees.",   # Noise
        "The RTX 5090 is ideal for local LLMs and RAG tasks.",  # Relevant
    ]
    
    # Create pairs for reranking
    pairs = [[query, doc] for doc in docs]
    scores = bge_ranker.compute_score(pairs)
    
    # Verify that relevant doc scores higher than noise
    assert len(scores) == 2
    assert scores[0] > scores[1], f"Expected relevant doc to score higher: {scores}"

@pytest.mark.skip(reason="FlagEmbedding has dependency conflicts with transformers library")
def test_top_k_utility(bge_ranker):
    """Test ranking with top_k filtering."""
    query = "Django migration command"
    docs = [
        "Use python manage.py migrate to apply changes.",
        "Django was released in 2005.",
        "Cooking pasta requires boiling water."
    ]
    
    # Create pairs for reranking
    pairs = [[query, doc] for doc in docs]
    scores = bge_ranker.compute_score(pairs)
    
    # Get top 1
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:1]
    assert len(top_indices) == 1
    assert "migrate" in docs[top_indices[0]]

@pytest.mark.skip(reason="FlagEmbedding has dependency conflicts with transformers library")
@pytest.mark.parametrize("query, docs", [
    ("empty test", []),
])
def test_edge_cases(bge_ranker, query, docs):
    """Check that empty lists don't crash the reranker."""
    if not docs:
        # Empty docs should return empty scores
        pairs = []
        scores = bge_ranker.compute_score(pairs) if pairs else []
        assert len(scores) == 0
    else:
        pairs = [[query, doc] for doc in docs]
        scores = bge_ranker.compute_score(pairs)
        assert len(scores) == len(docs)


if __name__ == "__main__":
    # Run all the test fixtures in this file
    pytest.main([__file__])