"""
This module contains functions to convert documents to text, for example, converting a .doc file to a .docx file,
a .docx file to a text file, or a .pdf file to a text file.
"""

# uv add python-docx pypdf rdflib striprtf
import io
import re
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
        text = '\n\n'.join([page.extract_text(x_tolerance=5, y_tolerance=3, layout=True, x_density=7.25, y_density=13) for page in pdf.pages])
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

def docx_bytes_to_markdown(b : bytes) -> str:
    document = Document.from_bytes(b)
    markdown_lines = []
    
    for paragraph in document.paragraphs:
        if not paragraph.text.strip():
            continue
        
        style = paragraph.style.name
        text = paragraph.text
        
        if style.startswith('Heading 1'):
            markdown_lines.append(f"# {text}")
        elif style.startswith('Heading 2'):
            markdown_lines.append(f"## {text}")
        elif style.startswith('Heading 3'):
            markdown_lines.append(f"### {text}")
        elif style.startswith('Heading 4'):
            markdown_lines.append(f"#### {text}")
        elif style.startswith('Heading 5'):
            markdown_lines.append(f"##### {text}")
        elif style.startswith('Heading 6'):
            markdown_lines.append(f"###### {text}")
        elif style.startswith('List'):
            level = paragraph.paragraph_format.left_indent
            indent = '  ' * (level // 914400) if level else 0  # 914400 twips = 1 inch
            markdown_lines.append(f"{indent}- {text}")
        else:
            markdown_lines.append(text)
    
    return '\n\n'.join(markdown_lines)




def xlsx_bytes_to_markdown(b : bytes) -> str:
    import pandas as pd
    import io

    # We cannot just convert this to a markdown table because the columns are not always aligned and there migth be too many columns.
    excel_file = pd.ExcelFile(io.BytesIO(b))
    markdown_parts = []
    
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(io.BytesIO(b), sheet_name=sheet_name)
        markdown_parts.append(f"## {sheet_name}\n")

        # For each row in the dataframe, print "ROW: {i}"
        #   For each column in the row, print "* {column}: {value}"
        for i, row in df.iterrows():
            markdown_parts.append(f"ROW: {i}")
            for column in df.columns:
                markdown_parts.append(f"* {column}: {row[column]}")
            markdown_parts.append("")
    
    return '\n'.join(markdown_parts)




def pptx_bytes_to_markdown(b : bytes) -> str:
    import io
    from pptx import Presentation

    presentation = Presentation(io.BytesIO(b))
    markdown_parts = []
    
    for slide_num, slide in enumerate(presentation.slides, 1):
        markdown_parts.append(f"## Slide {slide_num}\n")
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text.strip():
                markdown_parts.append(shape.text)
    
    return '\n\n'.join(markdown_parts)




def all_files_to_text(folder_path, cleaned_extension='.cleaned', overwrite=False, filter=None):
    "For each file F in folder_path, clean F, convert F to text, and save the text to G where G = F + cleaned_extension"
    def keep(x): return True
    if filter is None:
        filter = keep
    for F in glob.glob(os.path.join(folder_path, '*.*'), recursive=True):
        if F.endswith(cleaned_extension):
            continue
        G = F + cleaned_extension
        if os.path.exists(G) and not overwrite:
            continue
        # If the lastupdated time of F is greater than the lastupdated time of G, then overwrite G.
        write = False
        if os.path.exists(G):
            if os.path.getmtime(F) > os.path.getmtime(G):
                write = True
        else:
            write = True
                
        if write:
            text = get_text(F)
            # Replace all double spaces with single spaces
            lt = ""
            while lt != text:
                lt = text
                text = re.sub(r'\s+', ' ', text)
            if filter(text):
                writeText(G, text)
            else:
                if os.path.exists(G):
                    os.remove(G)
    pass




# ============================== TESTS ==============================

def test_convert_doc_to_docx():
    doc_file = r"..\data\corpus2\Niven, Larry - Unfinished Story.doc"
    extracted_text = convert_doc_to_docx(doc_file)
    print(extracted_text)

def test_convert_pdf_to_text():
    pdf_file = r"D:\rob\Wilmott Magazine\wilmott-202507-magazine-poulsen.pdf"
    extracted_text = get_text(pdf_file)
    print(extracted_text)


if __name__ == "__main__":
    # test_convert_doc_to_docx()
    test_convert_pdf_to_text()