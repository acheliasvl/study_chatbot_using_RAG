import fitz
from typing import List, Dict

def extract_text_and_metadata_from_pdf(pdf_path: str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    pages = []
    source = pdf_path.split('/')[-1].split('\\')[-1]
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({
                "text": text,
                "metadata": {
                    "source": source,
                    "page": page_num + 1,
                    "chapter": "unknown",
                    "section": "unknown"
                }
            })
    return pages

def chunk_text_with_context(pages: List[Dict], chunk_size: int, chunk_overlap: int) -> List[Dict]:
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            context_prefix = f"Source: {page['metadata']['source']}, Page: {page['metadata']['page']}\n\n"
            chunks.append({
                "text": context_prefix + chunk_text,
                "metadata": page["metadata"]
            })
            start += chunk_size - chunk_overlap
    return chunks
