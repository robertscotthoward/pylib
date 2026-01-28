# uv add langchain-text-splitters
import lib.tools
from lib.tools import *


class Reranker:
    def __init__(self):
        pass

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
        


if __name__ == "__main__":
    reranker = Flag_Reranker()