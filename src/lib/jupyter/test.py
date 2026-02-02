from bs4 import BeautifulSoup
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
import subprocess
import os
import re
from xhtml2pdf import pisa


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



def generate_notebook(filename="demo_notebook"):
    nb = nbf.v4.new_notebook()

    # 1. Add Markdown cell
    text = """# Data Analysis Report
This notebook was generated automatically using a Python script. 
Below is a demonstration of **pandas** data and **matplotlib** visualization."""
    nb['cells'].append(nbf.v4.new_markdown_cell(text))

    # 2. Add Code cell: Imports and Data
    code_data = """import pandas as pd
import matplotlib.pyplot as plt

# Create a sample dataset
data = {
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
    'Revenue': [1200, 1500, 1100, 1900, 2300],
    'Costs': [1000, 1100, 900, 1200, 1400]
}
df = pd.DataFrame(data)
print(df)"""
    nb['cells'].append(nbf.v4.new_code_cell(code_data))

    # 3. Add Code cell: Plotting
    code_plot = """plt.figure(figsize=(10, 5))
plt.plot(df['Month'], df['Revenue'], marker='o', label='Revenue')
plt.plot(df['Month'], df['Costs'], marker='s', label='Costs')
plt.title('Monthly Financial Performance')
plt.legend()
plt.grid(True)
plt.show()"""
    nb['cells'].append(nbf.v4.new_code_cell(code_plot))

    # 4. Execute the notebook to generate outputs
    print("Executing notebook...")
    try:
        ep = ExecutePreprocessor(timeout=60, kernel_name='python3')
        nb, resources = ep.preprocess(nb, {'metadata': {'path': os.getcwd()}})
        print("Notebook executed successfully")
    except Exception as e:
        print(f"Warning: Could not execute notebook: {e}")
        print("Continuing with unexecuted notebook...")

    # 5. Save the .ipynb file
    ipynb_file = f"{filename}.ipynb"
    with open(ipynb_file, 'w') as f:
        nbf.write(nb, f)
    print(f"Created {ipynb_file}")

    # 6. Convert to HTML
    html_exporter = HTMLExporter()
    (body, resources) = html_exporter.from_notebook_node(nb)

    html_file = f"{filename}.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(body)
    
    print(f"Successfully converted to {html_file}")
    
    return html_file, ipynb_file


def generate_notebook_and_pdf(filename="demo_notebook"):
    html_file, ipynb_file = generate_notebook(filename)

    with open(html_file, 'r', encoding='utf-8') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')

    # Target the high-level containers that hold BOTH the prompt and the code
    # 'jp-InputArea' is common in Lab, 'input' in classic Notebook
    classes_to_remove = [
        'prompt', 'input_prompt', 'output_prompt', 'out_prompt',
        'jp-InputArea', 'jp-InputPrompt', 'input', 'input_area'
    ]
    
    for div in soup.find_all("div", class_=classes_to_remove):
        div.decompose()

    # Special check: sometimes the 'In [x]:' is in a span or div without a unique class 
    # but inside a cell_mirror or input_area. 
    # Decomposing the classes above usually catches 99% of it.

    cleaned_html = str(soup)

    pdf_file = f"{filename}.html"
    with open(pdf_file, "w", encoding='utf-8') as f:
        f.write(cleaned_html)
    pdf_file = f"{filename}.pdf"
    html_to_pdf(cleaned_html, pdf_file)
    
    print(f"\nReport Cleaned and Created!")


if __name__ == "__main__":
    generate_notebook_and_pdf()