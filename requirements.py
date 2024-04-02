import sys
import subprocess

# List of library names to import
library_names = ['langchain-openai', 'openai', 'PyPDF2', 'tiktoken',
                 'faiss-cpu', 'textwrap', 'python-docx', 'python-pptx',
                 'ipykernel']

# Dynamically import libraries from list
for name in library_names:
    try:
        __import__(name)
    except ImportError:
        print(f"{name} not found. Installing {name}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', name])
