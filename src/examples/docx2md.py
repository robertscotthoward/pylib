import os 
import subprocess


inFolder = r"C:\Users\rober\Zinclusive\Zinclusive - Investors\5. Due Diligence FINAL"
outFolder = r"D:\rob\zinclusive\policies"


def convert_docx_to_md(inPath, outPath):
    cmd= f"""pandoc -s "{inPath}" -o "{outPath}" --from docx --to markdown"""
    subprocess.run(cmd, shell=True)


# For all docx files in the inFolder, convert them to md files in the outFolder
for file in os.listdir(inFolder):
    if file.endswith(".docx"):
        inPath = os.path.join(inFolder, file)
        outPath = os.path.join(outFolder, file.replace(".docx", ".md"))
        convert_docx_to_md(inPath, outPath)