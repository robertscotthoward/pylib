import os
import numpy as np
import joblib
from lib.ai.vectordb import VectorDb


class NumpyVectorDb(VectorDb):
    """Fast, local RAG system using NumPy for vector similarity."""
    
    def __init__(self, corpus, splitter, collection_path):
        super().__init__(corpus, splitter, os.path.basename(collection_path), None)
        self.collection_path = os.path.abspath(collection_path)
        if not os.path.exists(self.collection_path):
            os.makedirs(self.collection_path)
            
        self.index_file = os.path.join(self.collection_path, "vector_index.pkl")
        
        # Load the embedding model (Bi-Encoder)
        # Using a high-performance local model
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # In-memory storage
        self.vectors = None      # NumPy Matrix
        self.documents = []      # List of strings
        self.metadatas = []      # List of dicts
        self.ids = []           # List of strings
        
        self.load_database()

    def commit_batch(self, threshold=0):
        """Convert pending chunks into embeddings and merge into the NumPy index."""
        if not hasattr(self, 'chunk_batch') or not self.chunk_batch['chunks']:
            return

        if len(self.chunk_batch['chunks']) < threshold:
            return

        print(f"Embedding batch of {len(self.chunk_batch['chunks'])} chunks...")
        
        # Generate embeddings locally
        new_vectors = self.embedder.encode(self.chunk_batch['chunks'], convert_to_numpy=True)
        
        if self.vectors is None:
            self.vectors = new_vectors.astype('float32')
        else:
            self.vectors = np.vstack([self.vectors, new_vectors]).astype('float32')
            
        self.documents.extend(self.chunk_batch['chunks'])
        self.metadatas.extend(self.chunk_batch['metadatas'])
        self.ids.extend(self.chunk_batch['ids'])
        
        # Reset batch
        self.chunk_batch = {'chunks': [], 'metadatas': [], 'ids': []}
        self.save_database()

    def retrieve_documents(self, query, n_results=80):
        """Search NumPy index and rerank top results."""
        if self.vectors is None or len(self.vectors) == 0:
            return "No documents indexed."

        # 1. Embed Query
        query_vec = self.embedder.encode([query], convert_to_numpy=True)

        # 2. Fast NumPy Search (Cosine Similarity)
        # Normalizing vectors makes dot product equivalent to cosine similarity
        norms = np.linalg.norm(self.vectors, axis=1)
        q_norm = np.linalg.norm(query_vec)
        scores = np.dot(self.vectors, query_vec.T).flatten() / (norms * q_norm)

        # 3. Get Top N candidates
        top_indices = np.argsort(scores)[-n_results:][::-1]
        
        raw_docs = [self.documents[i] for i in top_indices]
        raw_ids = [self.ids[i] for i in top_indices]
        raw_metas = [self.metadatas[i] for i in top_indices]

        # 4. Rerank with Cross-Encoder if available
        reranker = self.get_reranker()
        if reranker:
            pairs = [[query, doc] for doc in raw_docs]
            rerank_scores = reranker.compute_score(pairs, batch_size=32)
            
            # Sort by reranker score
            ranked = sorted(zip(raw_docs, raw_ids, raw_metas, rerank_scores), 
                            key=lambda x: x[3], reverse=True)[:12]
        else:
            # No reranker - use original similarity scores
            ranked = sorted(zip(raw_docs, raw_ids, raw_metas, scores[top_indices]), 
                            key=lambda x: x[3], reverse=True)[:12]

        # 5. Format Output
        context_parts = []
        for doc, doc_id, meta, score in ranked:
            filename = meta.get('filename', 'Unknown')
            file_date = meta.get('file_date') or (doc_id.split('::')[0] if '::' in doc_id else None)
            date_str = f" (Date: {file_date})" if file_date else ""
            context_parts.append(f"From {filename}{date_str}:\n{doc}\n")
            
        return "\n---\n".join(context_parts)

    def save_database(self):
        """Serialize index to disk."""
        data = {
            'vectors': self.vectors,
            'documents': self.documents,
            'metadatas': self.metadatas,
            'ids': self.ids
        }
        joblib.dump(data, self.index_file)

    def load_database(self):
        """Load index from disk into memory."""
        if os.path.exists(self.index_file):
            data = joblib.load(self.index_file)
            self.vectors = data['vectors']
            self.documents = data['documents']
            self.metadatas = data['metadatas']
            self.ids = data['ids']
            print(f"Loaded {len(self.ids)} chunks from local index.")

    def get_embedded_files(self):
        """Return list of (filename, chunk_count) for deduplication."""
        from collections import Counter
        filenames = [m.get('filename') for m in self.metadatas]
        counts = Counter(filenames)
        return list(counts.items())

    def get_entry_count(self):
        """Return the number of entries (chunks) in this collection."""
        try:
            count = len(self.ids) if self.ids else 0
            print(f"[DEBUG] get_entry_count: returning {count}")
            return count
        except Exception as e:
            print(f"[ERROR] get_entry_count: {e}")
            import traceback
            traceback.print_exc()
            return 0




if __name__ == "__main__":
    collection_name = "corpus1"
    vdb_type = "numpy"
    corpus_folder = os.path.abspath(fr"C:\Rob\GitHub\robertscotthoward\python-ollama-example\data\test\{vdb_type}\{collection_name}")
    collection_path = os.path.abspath(f"data/vdb/{vdb_type}/{collection_name}")
    from lib.splitter import RecursiveCharacterText_Splitter
    splitter = RecursiveCharacterText_Splitter(chunk_size=1000, chunk_overlap=200)
    vdb = NumpyVectorDb(corpus_folder, splitter, collection_path)



    context = vdb.retrieve_documents("I want to shift into another plane. What spell should I use?")
    print(context)