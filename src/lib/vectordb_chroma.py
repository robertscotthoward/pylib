import json
import os
import sqlite3

import chromadb
from chromadb.config import Settings

from lib.corpus import Corpus
from lib.splitter import RecursiveCharacterText_Splitter
from lib.tools import *
from lib.vectordb import VectorDb


class ChromaVectorDb(VectorDb):
    """RAG system for querying a corpus using ChromaDB vector database"""

    def __init__(self, corpus, splitter, collection_path=None, reranker=None):
        super().__init__(corpus, splitter, os.path.basename(collection_path) if collection_path else None, reranker)
        # Suppose collection_path is D:\rob\rag\vectordb\zinweb
        # Then collection_dir is D:\rob\rag\vectordb\zinweb
        # And collection_name is zinweb
        # Each collection has a chroma_rag.json file in the collection_path
        self.collection_path = os.path.abspath(collection_path) if collection_path else None
        self.collection_dir = collection_path
        self.collection_name = os.path.basename(collection_path) if collection_path else None
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory=self.collection_dir
        ))
        
        # Get or create the collection
        if self.collection_name:
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            if self.collection.count():
                print(f"Collection {self.collection_name} loaded with {self.collection.count()} entries.")
            else:
                print(f"Collection {self.collection_name} created.")
    
    
    def commit_batch(self, threshold=0):
        if hasattr(self, 'chunk_batch') and self.chunk_batch and len(self.chunk_batch['chunks']) >= threshold:
            self.collection.add(
                documents=self.chunk_batch['chunks'],
                metadatas=self.chunk_batch['metadatas'],
                ids=self.chunk_batch['ids']
            )
            self.chunk_batch = None




    def retrieve_documents(self, query, n_results=80):
        raw_results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if not raw_results['documents'] or not raw_results['documents'][0]:
            return "No relevant documents found."

        docs = raw_results['documents'][0]
        ids = raw_results['ids'][0]
        metas = raw_results['metadatas'][0]

        with Spy('retrieve_documents') as spy:
            if self.reranker:
                # Batch processing is much faster than individual loops
                pairs = [[query, doc] for doc in docs]
                scores = self.reranker.compute_score(pairs, batch_size=32) 

                # 4. Sort and Slice in one pass
                # Using 'zip' and sorting is fast, but limit 'top_k' immediately
                ranked = sorted(zip(docs, ids, metas, scores), key=lambda x: x[3], reverse=True)[:12]

                # 5. Build context using list comprehension (faster than append loop)
                context_parts = []
                for doc, doc_id, meta, score in ranked:
                    filename = meta.get('filename', 'Unknown')
                    file_date = meta.get('file_date') or (doc_id.split('::')[0] if '::' in doc_id else None)
                    date_str = f" (Date: {file_date})" if file_date else ""
                    context_parts.append(f"From {filename}{date_str}:\n{doc}\n")
            else:
                from flashrank import Ranker, RerankRequest
                from pathlib import Path
                import tempfile

                # Initialize once outside your loop
                cache_dir = Path(tempfile.gettempdir()) / "flashrank_cache"
                cache_dir.mkdir(parents=True, exist_ok=True)                
                ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir=str(cache_dir))

                # Prepare data
                passages = [{"id": i, "text": doc, "meta": meta} for i, (doc, meta) in enumerate(zip(docs, metas))]
                rerankrequest = RerankRequest(query=query, passages=passages)

                results = ranker.rerank(rerankrequest)
                # Results are now ordered by score
                ranked = [(r['text'], r['id'], r['meta']) for r in results[:12]]
                context_parts = [f"From {meta.get('filename', 'Unknown')}:\n{doc}\n" for doc, doc_id, meta in ranked]

        return "\n---\n".join(context_parts)


    def get_embedded_files(self):
        "Return the number of documents (not just chunks) in this corpus."
        sqlitePath = os.path.join(self.collection_path, "chroma.sqlite3")
        print(f"[DEBUG] get_embedded_files: collection_path={self.collection_path}")
        print(f"[DEBUG] get_embedded_files: sqlitePath={sqlitePath}")
        print(f"[DEBUG] get_embedded_files: sqlite file exists={os.path.exists(sqlitePath)}")
        
        if not os.path.exists(sqlitePath):
            print(f"[WARNING] SQLite database not found at {sqlitePath}")
            print(f"[DEBUG] Contents of {self.collection_path}:")
            try:
                for item in os.listdir(self.collection_path):
                    item_path = os.path.join(self.collection_path, item)
                    print(f"  - {item} (dir={os.path.isdir(item_path)})")
            except Exception as e:
                print(f"[ERROR] Could not list directory: {e}")
            return []
        
        try:
            db = sqlite3.connect(sqlitePath)
            
            # First, let's check what the actual embedding_id format looks like
            cursor = db.execute("SELECT embedding_id FROM embeddings LIMIT 5")
            sample_ids = [row[0] for row in cursor.fetchall()]
            print(f"[DEBUG] get_embedded_files: sample embedding_ids: {sample_ids}")
            
            # Try the original query first
            sql = """
SELECT 
    SUBSTR(
        embedding_id, 
        INSTR(embedding_id, '::') + 2, 
        INSTR(embedding_id, '#') - (INSTR(embedding_id, '::') + 2)
    ) AS fp,
	count(*) N
FROM embeddings
WHERE INSTR(embedding_id, '::') > 0 AND INSTR(embedding_id, '#') > 0
group by 1
            """
            cursor = db.execute(sql)
            d = [row for row in cursor.fetchall()]
            print(f"[DEBUG] get_embedded_files: found {len(d)} unique documents with original query")
            
            # If no results, try alternative approach using metadata
            if len(d) == 0:
                print(f"[DEBUG] get_embedded_files: original query returned 0, trying metadata approach")
                sql_alt = """
SELECT 
    json_extract(embedding_id, '$.filename') AS filename,
    count(*) N
FROM embeddings
WHERE json_extract(embedding_id, '$.filename') IS NOT NULL
GROUP BY filename
                """
                cursor = db.execute(sql_alt)
                d = [row for row in cursor.fetchall()]
                print(f"[DEBUG] get_embedded_files: found {len(d)} unique documents with metadata query")
            
            db.close()
            print(f"[DEBUG] get_embedded_files: returning {len(d)} unique documents")
            return d
        except Exception as e:
            print(f"[ERROR] get_embedded_files: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_entry_count(self):
        """Return the number of entries (chunks) in this collection."""
        try:
            if self.collection:
                count = self.collection.count()
                print(f"[DEBUG] get_entry_count: returning {count}")
                return count
            else:
                print(f"[WARNING] get_entry_count: collection is None")
                return 0
        except Exception as e:
            print(f"[ERROR] get_entry_count: {e}")
            import traceback
            traceback.print_exc()
            return 0




