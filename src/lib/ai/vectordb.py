import os
import json
import datetime
import re
from lib.ai.fileconvert import get_text
from lib.ai.splitter import *
from lib.tools import *
from lib.ai.corpus import *
from chromadb.config import Settings




def extract_date_from_path(filepath):
    """
    Extract date in YYYY-MM-DD format from file path.
    Returns None if no date found.
    """
    # Pattern to match YYYY-MM-DD format
    date_pattern = r'\b(\d{4}-\d{2}-\d{2})\b'
    match = re.search(date_pattern, filepath)
    if match:
        return match.group(1)
    return None




def read_corpus_document(filepath):
    return get_text(filepath)




class VectorDb:
    """
    Base class for vector databases, which are objects that store and query embeddings of documents.
    @corpus is the corpus of documents to store and query.
    @splitter is the splitter to use to split the documents into chunks.
    @collection_name is the name of the collection to use.
    @reranker is the reranker to use to rerank the results.
    """
    def __init__(self, corpus, splitter, collection_name, reranker=None):
        self.collection_name = collection_name
        self.corpus = corpus
        self.splitter = splitter
        self.reranker = reranker

    def retrieve_documents(self, query, n_results=80):
        raise NotImplementedError("Subclasses must implement this method.")


    def add_chunk(self, chunk, filepath, chunk_index, file_date=None):
        if not hasattr(self, 'chunk_batch') or not self.chunk_batch:
            self.chunk_batch = {
                'chunks': [],
                'metadatas': [],
                'ids': []
            }
        self.chunk_batch['chunks'].append(chunk)
        metadata = {
            "filename": filepath, 
            "chunk_index": chunk_index
        }
        if file_date:
            metadata["file_date"] = file_date
        self.chunk_batch['metadatas'].append(metadata)
        # Include date in ID if available
        if file_date:
            self.chunk_batch['ids'].append(f"{file_date}::{filepath}#{chunk_index}")
        else:
            self.chunk_batch['ids'].append(f"{filepath}#{chunk_index}")


    def add_document(self, filepath):
        print(f"Adding document: {filepath} of size {os.path.getsize(filepath)} bytes")
        text = self.corpus.get_text(filepath)
        if not text:
            return
        chunks = self.splitter.get_chunks(text)
        if not chunks:
            return 0

        # Try to extract date from file path first (e.g., "2025-07-17" in path)
        file_date = extract_date_from_path(filepath)
        
        # If no date in path, fall back to file modification time
        if not file_date:
            file_mtime = os.path.getmtime(filepath)
            file_date = datetime.datetime.fromtimestamp(file_mtime).strftime("%Y-%m-%d")
        
        for chunk_index, chunk in enumerate(chunks):
            self.add_chunk(chunk, filepath, chunk_index, file_date=file_date)
        self.commit_batch(threshold=100)
        return len(chunks)


    def commit_batch(self, threshold=0):
        raise NotImplementedError("Subclasses must implement this method.")

    def save_database(self):
        """Save the database to disk. Subclasses can override if needed."""
        pass


    def get_embedded_files(self):
        raise NotImplementedError("Subclasses must implement this method.")


    def get_file_count(self):
        """Return the number of unique files (documents) in this corpus."""
        try:
            embedded_files = self.get_embedded_files()
            count = len(embedded_files)
            print(f"[DEBUG] get_file_count: returning {count}")
            return count
        except Exception as e:
            print(f"[ERROR] get_file_count: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def get_entry_count(self):
        """Return the number of entries (chunks) in this corpus. Subclasses should override."""
        raise NotImplementedError("Subclasses must implement this method.")


    def load_corpus(self, last_updated=0):
        """
        Load corpus from a folder into the vector database
        @corpus_folder is the folder to load the corpus from.
        @last_updated is the minimum last updated time of the files in the corpus to load. All other files will be skipped. If 0, all files will be loaded.
        @return the maximum last updated time of the files in the corpus.
        """
        
        # If no corpus folder is configured, skip loading
        if not self.corpus.corpus_folder:
            print(f"[DEBUG] load_corpus: No corpus folder configured, skipping load")
            return last_updated

        fnChunkless = os.path.join(self.collection_path, 'chunkless.json')
        if os.path.exists(fnChunkless):
            chunkless = readJson(fnChunkless)
            chunkless = chunkless.get('chunkless', [])
            chunkless = set(chunkless)
        else:
            chunkless = set()

        # Pre-scan the corpus to find the maximum last updated time
        m = 0
        num_documents = 0
        for filepath in self.corpus.enumerate_files():
            num_documents += 1
            file_updated = os.path.getmtime(filepath)
            if file_updated > m:
                m = file_updated
        sMaxUpdate = datetime.datetime.fromtimestamp(m).isoformat()
        self.num_documents = num_documents

        # if m == last_updated and numDocs == self.corpus.get_file_count():
        #     return m

        embedded_files = self.get_embedded_files()
        embedded_filenames = [row[0] for row in embedded_files]
        embedded_filenames = set(embedded_filenames)
        redo_files = set([row[0] for row in embedded_files if row[1] < 100])
        redo_files = set()
        self.file_count_before = embedded_files
        max_updated = 0
        for filepath in self.corpus.enumerate_files():
            # Only add the file if it does not already exists in the vectordb, or if it has changed since last run.
            file_updated = os.path.getmtime(filepath)
            
            should_add = False
            ext = os.path.splitext(filepath)[1]
            if filepath in redo_files and ext in ['.pdf']:
                should_add = True
            elif not filepath in embedded_filenames:
                # File doesn't exist in vectordb - add it
                should_add = True
            else:
                # File exists in vectordb - only add if it has been updated since last run
                if file_updated > m:
                    should_add = True
            
            # Track the maximum updated time across all files
            if file_updated > max_updated:
                max_updated = file_updated
                
            if filepath in chunkless:
                should_add = False

            if should_add:
                n_chunks = self.add_document(filepath)
                if not n_chunks:
                    chunkless.add(filepath)
                    continue

        writeJson(fnChunkless, {'chunkless': list(chunkless)})

        self.commit_batch()
        embedded_files = set(self.get_embedded_files())
        self.file_count_after = len(embedded_files)
        return max_updated
