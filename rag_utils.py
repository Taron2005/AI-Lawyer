import fitz  # PyMuPDF
from typing import List

def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """
    Extracts text from a file based on its extension.
    Supports .pdf and .txt files.
    """
    ext = f".{filename.split('.')[-1].lower()}"
    text = ""
    try:
        if ext == ".pdf":
            with fitz.open(stream=file_content, filetype="pdf") as doc:
                text = "".join(page.get_text() for page in doc)
        elif ext == ".txt":
            text = file_content.decode("utf-8")
        return text
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
        return ""

def smart_chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> List[str]:
    """
    Splits text into small, overlapping chunks to ensure no loss of context
    at chunk boundaries. Adjusted for slightly larger chunks.
    """
    if not text:
        return []
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks
