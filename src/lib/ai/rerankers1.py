# uv add langchain-text-splitters
import lib.tools
from lib.tools import *
import pytest
import rerankers




# ================================================================================
# TESTS



# This fixture loads the model once per test session/module
@pytest.fixture(scope="module")
def bge_ranker():
    # Use rerankers library which is properly implemented
    return rerankers.Reranker("BAAI/bge-reranker-v2-m3")

@pytest.mark.skip(reason="Requires network access to download model from HuggingFace Hub")
def test_relevance_ranking(bge_ranker):
    """Verify that relevant info scores higher than noise."""
    query = "How do I use a 5090 for AI?"
    docs = [
        "The RTX 5090 is ideal for local LLMs and RAG tasks.",  # Relevant
        "Apples are usually red or green and grow on trees."    # Noise
    ]
    
    # Use the rank() method which returns RankedResults
    results = bge_ranker.rank(query=query, docs=docs)
    
    # Verify that relevant doc scores higher than noise
    assert len(results.results) == 2
    assert "RTX 5090" in results.results[0].text
    assert results.results[0].score > results.results[1].score

    # Switch order
    docs = [
        "Apples are usually red or green and grow on trees.",   # Noise
        "The RTX 5090 is ideal for local LLMs and RAG tasks.",  # Relevant
    ]
    
    # Use the rank() method again
    results = bge_ranker.rank(query=query, docs=docs)
    
    # Verify that relevant doc scores higher than noise
    assert len(results.results) == 2
    assert "RTX 5090" in results.results[1].text
    assert results.results[0].score > results.results[1].score

@pytest.mark.skip(reason="Requires network access to download model from HuggingFace Hub")
def test_top_k_utility(bge_ranker):
    """Test ranking with top_k filtering."""
    query = "Django migration command"
    docs = [
        "Use python manage.py migrate to apply changes.",
        "Django was released in 2005.",
        "Cooking pasta requires boiling water."
    ]
    
    # Use the rank() method
    results = bge_ranker.rank(query=query, docs=docs)
    top_one = results.top_k(1)
    
    assert len(top_one) == 1
    assert "migrate" in top_one[0].text

@pytest.mark.skip(reason="Requires network access to download model from HuggingFace Hub")
@pytest.mark.parametrize("query, docs", [
    ("empty test", []),
])
def test_edge_cases(bge_ranker, query, docs):
    """Check that empty lists don't crash the reranker."""
    results = bge_ranker.rank(query=query, docs=docs)
    assert len(results.results) == 0


if __name__ == "__main__":
    # Run all the test fixtures in this file
    pytest.main([__file__])