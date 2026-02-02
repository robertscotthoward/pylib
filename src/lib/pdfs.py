import re
from xhtml2pdf import pisa

def clean_html_for_pdf(html):
    """
    Applies a professional, generic CSS template compatible with xhtml2pdf.
    Supports headers, footers, and page numbering.
    """
    # Remove any existing complex styles that might break the parser
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
    
    # Professional PDF Template
    generic_style = """
    <style>
        @page {
            size: letter;
            margin: 2cm;
            @frame footer {
                -pdf-frame-content: footerContent;
                bottom: 1cm;
                margin-left: 2cm;
                margin-right: 2cm;
                height: 1cm;
            }
        }
        
        body {
            font-family: Helvetica, Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.5;
            color: #222;
        }

        /* Typography */
        h1 { font-size: 24pt; color: #1a365d; border-bottom: 2px solid #1a365d; padding-bottom: 5px; margin-bottom: 20px; }
        h2 { font-size: 18pt; color: #2c5282; margin-top: 25px; border-bottom: 1px solid #e2e8f0; }
        h3 { font-size: 14pt; color: #4a5568; margin-top: 15px; }
        
        /* Lists & Tables */
        ul { margin-left: 20px; }
        li { margin-bottom: 8px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th { background-color: #edf2f7; color: #2d3748; font-weight: bold; padding: 10px; border: 1px solid #cbd5e0; }
        td { padding: 8px; border: 1px solid #e2e8f0; vertical-align: top; }
        
        /* Elements */
        .highlight { color: #2b6cb0; font-weight: bold; }
        .footer-text { text-align: center; font-size: 9pt; color: #718096; }
        
        /* Helpers */
        .pdf-page-break { -pdf-keep-with-next: true; }
    </style>
    """
    
    # Add a hidden footer element that @page can reference for page numbering
    footer_html = """
    <div id="footerContent" class="footer-text">
        Page <pdf:pagenumber> of <pdf:pagecount>
    </div>
    """

    # Assemble the document
    if '<body' in html:
        # Inject style into head, and footer into the start of body
        html = html.replace('</head>', generic_style + '</head>', 1)
        html = html.replace('<body', '<body' + footer_html, 1)
    else:
        html = f"<html><head>{generic_style}</head><body>{footer_html}{html}</body></html>"
    
    return html

def html_to_pdf(html_content, output_filename):
    """Convert HTML to PDF using xhtml2pdf with improved error handling."""
    try:
        cleaned_html = clean_html_for_pdf(html_content)
        
        with open(output_filename, "wb") as f:
            pisa_status = pisa.CreatePDF(cleaned_html, dest=f)
        
        if not pisa_status.err:
            print(f"Successfully created: {output_filename}")
            return output_filename
        else:
            print(f"Conversion Error: {pisa_status.err}")
            return None
    except Exception as e:
        print(f"System Error: {e}")
        return None

def markdown_to_pdf(markdown_content, output_filename):
    """Convert Markdown to PDF via markdown2."""
    import markdown2
    # Use 'extras' to support tables and task lists in markdown
    html_content = markdown2.markdown(markdown_content, extras=["tables", "fenced-code-blocks"])
    return html_to_pdf(html_content, output_filename)