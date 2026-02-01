def clean_html_for_pdf(html):
    """Remove or simplify CSS that xhtml2pdf can't handle."""
    # Remove style tags with complex CSS selectors
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
    
    # Add basic styling
    basic_style = """
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2, h3 { color: #333; }
        pre { background-color: #f5f5f5; padding: 10px; overflow-x: auto; }
        code { font-family: monospace; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }

        /* Hide the prompt area (the "In [x]:" part) */
        .prompt, .input_prompt, .output_prompt {
            display: none;
            visibility: hidden;
            width: 0;
        }

        /* Optional: Remove the left padding that usually holds the prompt */
        div.input_area {
            border: 1px solid #cfcfcf;
            border-radius: 2px;
            background: #f7f7f7;
            line-height: 1.21429em;
        }
    </style>

    """
    
    # Insert basic style after opening body tag or at the beginning
    if '<body' in html:
        html = html.replace('<body', basic_style + '<body', 1)
    else:
        html = basic_style + html
    
    return html


def html_to_pdf(html_content, output_filename):
    """Convert HTML to PDF using xhtml2pdf."""
    try:
        # Clean the HTML
        html_content = clean_html_for_pdf(html_content)
        
        with open(output_filename, "w+b") as f:
            pisa_status = pisa.CreatePDF(html_content, dest=f)
        
        if pisa_status.err:
            print(f"Error converting HTML to PDF: {pisa_status.err}")
            return None
        
        print(f"Successfully converted to {output_filename}")
        return output_filename
    except Exception as e:
        print(f"Error converting HTML to PDF: {e}")
        return None



