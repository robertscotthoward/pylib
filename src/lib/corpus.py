import os
import lib.fileconvert
from lib.fileconvert import convert_all_doc_to_docx, docx_to_text, pdf_to_text
from lib.tools import *


file_extensions = [".docx", ".pdf", ".txt", ".md", ".rst", ".json", ".html", ".epub"]


class Corpus:
    """Class for enumerating a corpus of documents and getting the text of the documents."""
    

    def __init__(self, file_extensions=file_extensions, corpus_folder=None):
        self.file_extensions = file_extensions
        self.corpus_folder = corpus_folder


    def enumerate_files(self):
        """Load corpus from a folder into the vector database"""
        if not self.corpus_folder:
            return
        if not os.path.exists(self.corpus_folder):
            raise ValueError(f"Corpus folder not found: {self.corpus_folder}")

        for root, dirs, files in os.walk(self.corpus_folder):
            for file in files:
                if file.endswith(tuple(self.file_extensions)):
                    filepath = os.path.join(root, file)
                    filepath = filepath.replace('\\', '/')
                    yield filepath


    def get_text(self, filepath):
        return lib.fileconvert.get_text(filepath)


    def convert_files(self):
        convert_all_doc_to_docx(self.corpus_folder)


