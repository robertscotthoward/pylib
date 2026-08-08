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
from lib.tools import *
import pandas as pd





def needs_conversion(source_path, target_path) -> bool:
    """True when target is missing or older than source, i.e. the target is stale.

    Equal timestamps count as up to date, so a target stamped with its source's
    mtime (see convert_doc_to_docx) is not rebuilt on every run.
    """
    if not os.path.exists(target_path):
        return True
    if not os.path.exists(source_path):
        return False
    return os.path.getmtime(source_path) > os.path.getmtime(target_path)


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




def get_markdown(filepath):
    """Get the text of a file. Only specific file extensions are supported."""

    try:
        if not os.path.exists(filepath):
            return None
        if filepath.endswith(".pdf"):
            return pdf_bytes_to_markdown(readBytes(filepath))
        if filepath.endswith(".docx"):
            return docx_bytes_to_markdown(readBytes(filepath))
        if filepath.endswith(".rtf"):
            return rtf_to_text(filepath)
        if filepath.endswith(".rdf"):
            return rdf_to_text(filepath)
        if filepath.endswith(".epub"):
            return epub_to_markdown(filepath)
        if filepath.endswith(".xlsx"):
            return xlsx_bytes_to_markdown(readBytes(filepath))
        return readText(filepath)
    except Exception as e:
        print(f"Error getting text from {filepath}: {e}")
        return None



def epub_to_markdown(filepath):
    import pypandoc
    return pypandoc.convert_file(filepath, 'md', format='epub')


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




def pdf_bytes_to_text(bytes: bytes) -> str:
    import pdfplumber
    with pdfplumber.open(io.BytesIO(bytes)) as pdf:
        text = '\n\n'.join([page.extract_text(x_tolerance=5, y_tolerance=3, layout=True, x_density=7.25, y_density=13) for page in pdf.pages])
    return text


def _ensure_tessdata_prefix():
    """Point PyMuPDF's built-in OCR at Tesseract's language data, so scanned/image-only
    PDFs (no text layer) get OCR'd instead of silently producing empty markdown.
    """
    if os.environ.get('TESSDATA_PREFIX'):
        return
    try:
        config = getYaml('config.yaml') or {}
        tessdata_path = g(config, 'all/tessdata_path')
    except Exception:
        tessdata_path = None
    if not tessdata_path or not os.path.exists(tessdata_path):
        candidates = glob.glob(r"C:\Program Files\Tesseract-OCR\tessdata") + \
                     glob.glob(r"C:\Program Files (x86)\Tesseract-OCR\tessdata")
        tessdata_path = candidates[0] if candidates else None
    if tessdata_path and os.path.exists(tessdata_path):
        os.environ['TESSDATA_PREFIX'] = tessdata_path


def pdf_bytes_to_markdown(bytes: bytes) -> str:
    import pymupdf4llm
    import pymupdf
    _ensure_tessdata_prefix()
    pdf_stream = io.BytesIO(bytes)
    doc = pymupdf.open(stream=pdf_stream, filetype="pdf")
    return pymupdf4llm.to_markdown(doc)



def pdf_to_text(filepath):
    # import pypdf
    # pages = [page.extract_text() for page in pypdf.PdfReader(filepath).pages]
    # text = '\n\n'.join(pages)
    b = readBytes(filepath)
    return pdf_bytes_to_text(b)




def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    import pdfplumber
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = '\n\n'.join([page.extract_text(x_tolerance=5, y_tolerance=3, layout=True, x_density=7.25, y_density=13) for page in pdf.pages])
    return text




def convert_doc_to_docx(inPath, outPath=None):
    # Resolve Wordconv.exe: config.yaml > glob search
    wordconv_path = None
    try:
        config = getYaml('config.yaml') or {}
        wordconv_path = g(config, 'all/wordconv_path')
    except Exception:
        pass
    if not wordconv_path or not os.path.exists(wordconv_path):
        candidates = glob.glob(r"C:\Program Files\Microsoft Office*\**\Wordconv.exe", recursive=True)
        wordconv_path = candidates[0] if candidates else None
    if not wordconv_path or not os.path.exists(wordconv_path):
        print(f"❌ FILE NOT FOUND: Wordconv.exe not found. Set 'all/wordconv_path' in config.yaml.")
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
    from docx.table import Table

    document = Document(docx_path)
    parts = []
    for block in _iter_block_items(document):
        if isinstance(block, Table):
            lines = _table_to_markdown(block)
            if lines:
                parts.append('\n'.join(lines))
        elif block.text.strip():
            parts.append(block.text)
    return '\n\n'.join(parts)


def xls_bytes_to_markdown(byte_data):
    # 1. Wrap the byte array in a file-like object
    byte_stream = io.BytesIO(byte_data)
    
    # 2. Read the XLS file
    # Note: xlrd is required for .xls files
    df = pd.read_excel(byte_stream, engine='xlrd')
    
    # 3. Convert to Markdown
    # 'tabulate' is used under the hood for clean formatting
    return df.to_markdown(index=False)



def _iter_block_items(parent):
    """Yield Paragraph and Table objects from a document body or table cell, in document order.

    python-docx exposes `document.paragraphs` and `document.tables` as separate flat lists,
    which loses ordering and silently drops paragraphs nested inside table cells.
    """
    from docx.document import Document as _Document
    from docx.oxml.ns import qn
    from docx.table import Table, _Cell
    from docx.text.paragraph import Paragraph

    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError(f"Cannot iterate block items of {type(parent)}")

    for child in parent_elm.iterchildren():
        if child.tag == qn('w:p'):
            yield Paragraph(child, parent)
        elif child.tag == qn('w:tbl'):
            yield Table(child, parent)


def _unique_row_cells(row):
    """Return a row's cells with horizontally/vertically merged duplicates collapsed."""
    try:
        cells = list(row.cells)
    except (IndexError, ValueError):
        # Irregular grids can break python-docx's cell mapping; fall back to raw <w:tc>.
        from docx.table import _Cell
        cells = [_Cell(tc, row.table) for tc in row._tr.tc_lst]

    unique = []
    seen = set()
    for cell in cells:
        key = id(cell._tc)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cell)
    return unique


def _cell_content(cell, indent):
    """Return (inline_text, nested_table_lines) for a table cell."""
    from docx.table import Table

    texts = []
    nested = []
    for block in _iter_block_items(cell):
        if isinstance(block, Table):
            nested.extend(_table_to_markdown(block, indent + '  '))
        elif block.text.strip():
            texts.append(' '.join(block.text.split()))
    return ' '.join(texts), nested


def _table_to_markdown(table, indent=''):
    """Render a table as indented bullets: two-cell rows become `label: value`."""
    lines = []
    for row in table.rows:
        texts = []
        nested = []
        for cell in _unique_row_cells(row):
            text, cell_nested = _cell_content(cell, indent)
            texts.append(text)
            nested.extend(cell_nested)

        while texts and not texts[-1]:
            texts.pop()

        if len(texts) == 2:
            lines.append(f"{indent}- {texts[0]}: {texts[1]}")
        elif texts:
            lines.append(f"{indent}- " + ' | '.join(texts))
        lines.extend(nested)
    return lines


def _para_list_level(paragraph) -> int | None:
    """Return 0-based list indent level from XML numbering, or None if not a list paragraph."""
    from docx.oxml.ns import qn
    pPr = paragraph._p.find(qn('w:pPr'))
    if pPr is None:
        return None
    numPr = pPr.find(qn('w:numPr'))
    if numPr is None:
        return None
    ilvl = numPr.find(qn('w:ilvl'))
    if ilvl is None:
        return None
    try:
        return int(ilvl.get(qn('w:val'), 0))
    except (TypeError, ValueError):
        return None


def _paragraph_to_markdown(paragraph) -> str | None:
    """Render a single paragraph as markdown, or None when it is empty."""
    if not paragraph.text.strip():
        return None
    text = paragraph.text
    style = paragraph.style.name if paragraph.style else ''

    for n in range(1, 7):
        if style.startswith(f'Heading {n}'):
            return f"{'#' * n} {text}"

    level = _para_list_level(paragraph)
    if level is None and style.startswith('List'):
        level = 0
    if level is not None:
        return f"{'  ' * level}- {text}"
    return text


def docx_bytes_to_markdown(b : bytes) -> str | None:
    from docx.table import Table

    try:
        document = Document(io.BytesIO(b))
    except Exception as e:
        print(f"[ERROR] Failed to convert docx to markdown: {e}")
        return None
    markdown_lines = []

    for block in _iter_block_items(document):
        if isinstance(block, Table):
            lines = _table_to_markdown(block)
            if lines:
                markdown_lines.append('\n'.join(lines))
        else:
            md = _paragraph_to_markdown(block)
            if md is not None:
                markdown_lines.append(md)

    return '\n\n'.join(markdown_lines)




def doc_bytes_to_markdown(b : bytes) -> str:
    """
    Convert old Word (.doc) files to markdown text.
    Note: .doc files are binary format. This uses python-docx which has limited support.
    """
    return None
    inFile = r"cache/temp.doc"
    outFile = r"cache/temp.docx"
    writeBytes(inFile, b)
    convert_doc_to_docx(inFile, outFile)
    b =  readBytes(outFile)
    return docx_bytes_to_markdown(b)
    


def xlsx_bytes_to_markdown(b : bytes) -> str:
    import pandas as pd
    import io

    # We cannot just convert this to a markdown table because the columns are not always aligned and there migth be too many columns.
    excel_file = pd.ExcelFile(io.BytesIO(b))
    markdown_parts = []

    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(io.BytesIO(b), sheet_name=sheet_name)
        markdown_parts.append(f"# {sheet_name}")
        markdown_parts.append(df.to_csv(index=False, lineterminator='\n').strip('\n'))

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




def ppt_bytes_to_markdown(b : bytes) -> str:
    """
    Convert old PowerPoint (.ppt) files to markdown text.
    Note: .ppt files are binary format and harder to parse than .pptx.
    This uses a basic extraction approach.
    """
    try:
        # Try using python-pptx with a workaround for older formats
        # If that fails, fall back to basic text extraction
        import io
        from pptx import Presentation
        
        try:
            presentation = Presentation(io.BytesIO(b))
            markdown_parts = []
            
            for slide_num, slide in enumerate(presentation.slides, 1):
                markdown_parts.append(f"## Slide {slide_num}\n")
                for shape in slide.shapes:
                    if shape.has_text_frame and shape.text.strip():
                        markdown_parts.append(shape.text)
            
            return '\n\n'.join(markdown_parts)
        except (KeyError, Exception):
            # If python-pptx fails, try basic binary text extraction
            text_content = []
            # Look for readable text in the binary data
            try:
                decoded = b.decode('utf-16-le', errors='ignore')
                # Extract text between common delimiters
                import re
                text_matches = re.findall(r'[\x20-\x7E]{4,}', decoded)
                text_content = [t.strip() for t in text_matches if t.strip() and len(t) > 3]
            except:
                pass
            
            if text_content:
                return '\n'.join(text_content)
            else:
                return "Could not extract text from .ppt file. The file may be corrupted or in an unsupported format."
    
    except Exception as e:
        print(f"Error converting PPT: {e}")
        return f"Error converting PPT file: {e}"


def all_files_to_text(folder_path, cleaned_extension='.cleaned', overwrite=False, filter=None):
    "For each file F in folder_path, clean F, convert F to text, and save the text to G where G = F + cleaned_extension"
    def keep(x): return True
    if filter is None:
        filter = keep
    for F in glob.glob(os.path.join(folder_path, '*.*'), recursive=True):
        if F.endswith(cleaned_extension):
            continue
        G = F + cleaned_extension
        # Rewrite G when it is missing or older than F.
        if overwrite or needs_conversion(F, G):
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