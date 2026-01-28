import glob
import shutil
from lib.corpus import get_text
from lib.fileconvert import convert_doc_to_docx
from lib.tools import *



def transformDocToDocx(inputFolders, outputFolder):
    from lib.fileconvert import transform_all_doc_to_docx
    for inputFolder in inputFolders:
        transform_all_doc_to_docx(inputFolder, outputFolder)


def transformText(inputFolders, outputFolder):
    ensureFolder(outputFolder)
    for inputFolder in inputFolders:
        for fpIn in glob.glob(os.path.join(inputFolder, '**', '*.*'), recursive=True):
            fpOut = os.path.join(outputFolder, os.path.basename(fpIn))
            if not os.path.isfile(fpIn):
                continue
            text = readText(fpIn)
            writeText(fpOut, text)


def mapFolders(inputFolder, outputFolder, mapFunction=None):
    """
    Get all files recursively from the input folder and yield the input and output paths.
    The mapFunction is a function that takes the input file and returns the output file.
    If the output file already exists, it is not yielded.
    """
    ensureFolder(outputFolder)
    def MapFunction(filepath):
        return filepath.replace(inputFolder, outputFolder)
    if not mapFunction:
        mapFunction = MapFunction
    for inputPath in glob.glob(os.path.join(inputFolder, '**', '*.*'), recursive=True):
        if not os.path.isfile(inputPath):
            continue
        outputPath = mapFunction(inputPath)
        if not outputPath or os.path.exists(outputPath):
            continue
        ensureFolder(os.path.dirname(outputPath))
        yield inputPath, outputPath



class Workflow:
    def __init__(self, workflow):
        """
        @workflow: A dictionary containing the workflow configuration.
        """
        self.workflow = workflow
        self.root = self.workflow.get('root', '')


    def execute(self, task, inputFolders, outputFolder):
        return True


    def run(self):
        """
        Run the workflow.
        Determine the stateId from the workflow configuration.
        """

        # Create an ordered list of states to run in the order of least dependencies to most dependencies.
        stateKeys = []
        def addStateKey(stateKey, visited=set()):
            if stateKey in visited:
                return
            visited.add(stateKey)
            state = self.workflow['states'][stateKey]
            for pKey in state.get('prerequisites', []):
                addStateKey(pKey, visited)
            stateKeys.append(stateKey)
        for stateKey in self.workflow['states'].keys():
            addStateKey(stateKey)
        

        # Run the states in the order of the todoKeys until none are left that can be run, or an error occurs.
        for stateKey in stateKeys:
            state = self.workflow['states'][stateKey]
            outputFolder = os.path.join(self.root, stateKey)
            input = state.get('input', {})
            if input and 'folders' in input:
                inputFolder = input.get('folders', [])
            else:
                inputFolder = [os.path.join(self.root, f) for f in state.get('prerequisites', [])]

            # For each task in this state input folder that does not have a corresponding output file, run the task.
            ensureFolder(outputFolder)
            def mapFunction(inputPath):
                return inputPath.replace(inputFolder, outputFolder)
            for inputPath, outputPath in mapFolders(inputFolders, outputFolder):
                if os.path.exists(outputPath):
                    continue
                tasks = state.get('tasks', [])
                if not tasks:
                    print(f"No tasks found for state {stateKey}")
                    continue

                for task in tasks:
                    assert type(task) == dict
                    assert len(task) == 1
                    key = list(task.keys())[0].upper()
                    value = task[key]
                    if key == 'DO':
                        if not value in globals():
                            raise ValueError(f"Function {value} not found")
                        globals()[value](inputFolders, outputFolder)
                    else:
                        raise ValueError(f"Unknown task type: {key}")







def test1():
    for inputPath, outputPath in mapFolders(r"C:\Rob\RAG\Resumes, Work History, Career", r"C:\temp\output"):
        if inputPath.endswith(('.txt', '.md', '.rst', '.text')):
            shutil.copy2(inputPath, outputPath)
            print(f"Processing {inputPath}\n        -> {outputPath}")
        continue

        if inputPath.endswith('.doc'):
            outputPath = outputPath.replace('.doc', '.docx')
            convert_doc_to_docx(inputPath, outputPath)
        elif inputPath.endswith(('.txt', '.md', '.rst', '.text')):
            shutil.copy2(inputPath, outputPath)
            pass
        elif inputPath.endswith('.pdf'):
            text = get_text(inputPath)
            writeText(outputPath, text)
        else:
            print(f"Skipped: {inputPath}")

if __name__ == "__main__":
    test1()