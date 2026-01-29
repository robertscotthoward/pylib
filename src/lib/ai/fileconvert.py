"""
This module contains functions to convert documents to text, for example, converting a .doc file to a .docx file,
a .docx file to a text file, or a .pdf file to a text file.
"""

# uv add python-docx pypdf rdflib striprtf
from docx import Document
from lib.tools import ensureFolder, readText, writeText
import ebooklib
import glob
import os
import subprocess




def get_text(filepath):
    """Get the text of a file. Only specific file extensions are supported."""

    try:
        if not os.path.exists(filepath):
            return None
        if filepath.endswith(".pdf"):
            return pdf_to_text(filepath)
        if filepath.endswith(".docx"):
            return docx_to_text(filepath)
        if filepath.endswith(".rtf"):
            return rtf_to_text(filepath)
        if filepath.endswith(".rdf"):
            return rdf_to_text(filepath)
        if filepath.endswith(".epub"):
            return epub_to_text(filepath)
        return readText(filepath)
    except Exception as e:
        print(f"Error getting text from {filepath}: {e}")
        return None



def epub_to_text(filepath):
    from ebooklib import epub
    from bs4 import BeautifulSoup
    import re
    
    book = epub.read_epub(filepath)
    texts = []
    
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        # Get the content (HTML) from the EPUB item
        content = item.get_content()
        # Parse HTML and extract text
        soup = BeautifulSoup(content, 'html.parser')
        
        # Use space as separator to avoid breaking words
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        if text:
            texts.append(text)
    
    text = '\n\n'.join(texts)
    return text




def rdf_to_text(filepath):
    import rdflib
    graph = rdflib.Graph()
    graph.parse(filepath, format='turtle')
    return graph.serialize(format='turtle') # Serialize to Turtle (very readable)




def rtf_to_text(filepath):
    from striprtf.striprtf import rtf_to_text
    s = readText(filepath)
    plain_text = rtf_to_text(s)
    return plain_text




def pdf_to_text(filepath):
    # import pypdf
    # pages = [page.extract_text() for page in pypdf.PdfReader(filepath).pages]
    # text = '\n\n'.join(pages)
    import pdfplumber
    with pdfplumber.open(filepath) as pdf:
        text = '\n\n'.join([page.extract_text() for page in pdf.pages])
    return text




def convert_doc_to_docx(inPath, outPath=None):
    wordconv_path = r"C:\Program Files\Microsoft Office\root\Office16\Wordconv.exe"
    if not os.path.exists(wordconv_path):
        print(f"❌ FILE NOT FOUND: Wordconv not found at '{wordconv_path}'")
        return None

    if not os.path.exists(inPath):
        print(f"Error: Source file not found at '{inPath}'")
        return None
        
    inPath = inPath.replace('\\', '/')
    if outPath:
        outPath = outPath.replace('\\', '/')
    else:
        outDir = os.path.dirname(inPath)
        base_name = os.path.splitext(os.path.basename(inPath))[0]
        outPath = os.path.join(outDir, f"{base_name}.docx")
    
    # if os.path.exists(docx_path):
    #     # Set the last updated time to the doc file
    #     os.utime(docx_path, (os.path.getatime(doc_path), os.path.getmtime(doc_path)))

    if not os.path.exists(outPath):
        try:
            outPath = outPath.replace('\\', '/')
            ensureFolder(os.path.dirname(outPath))
            print(f"Converting '{inPath}' to text using docx...")
            subprocess.run([
                wordconv_path,
                "-oice",
                "-nme",
                inPath,
                outPath
            ], check=True, cwd=os.path.dirname(outPath), capture_output=True)

            # Set the last updated time to the docx file
            os.utime(outPath, (os.path.getatime(inPath), os.path.getmtime(inPath)))
           
        except FileNotFoundError:
            print("❌ CONVERSION FAILED: 'Wordconv' command not found. Ensure Pandoc is installed.")
            return None
        except subprocess.CalledProcessError as e:
            print(f"❌ CONVERSION FAILED: Wordconv error. Output: {e.stderr.decode()}")
            return None




def convert_all_doc_to_docx(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".doc"):
                convert_doc_to_docx(os.path.join(root, file))




def transform_all_doc_to_docx(inFolder, outFolder):
    for filepath in glob.glob(os.path.join(inFolder, '*.doc'), recursive=True):
        convert_doc_to_docx(filepath)




def docx_to_text(docx_path):
    document = Document(docx_path)
    text = '\n\n'.join(p.text for p in document.paragraphs)
    return text




def all_files_to_text(folder_path, cleaned_extension='.cleaned', overwrite=False):
    "For each file F in folder_path, clean F, convert F to text, and save the text to G where G = F + cleaned_extension"
    for F in glob.glob(os.path.join(folder_path, '*.*'), recursive=True):
        if F.endswith(cleaned_extension):
            continue
        G = F + cleaned_extension
        if os.path.exists(G) and not overwrite:
            continue
        text = get_text(F)
        writeText(G, text)



def test1():
    doc_file = r"..\data\corpus1\Niven, Larry - Unfinished Story.doc"
    extracted_text = convert_doc_to_docx(doc_file)
    print(extracted_text)

if __name__ == "__main__":
    test1()