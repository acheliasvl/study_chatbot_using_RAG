import os

import fitz  # PyMuPDF


def extract_pages(pdf_path: str) -> list[dict]:
    """PDF -> one dict per non-empty page: text + source + page number."""
    source = os.path.basename(pdf_path)
    pages = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                pages.append({"text": text, "metadata": {"source": source, "page": page_num}})
    return pages


def chunk_pages(pages: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """Split each page into overlapping chunks ("contextual chunking").

    Every chunk starts with 'Source: <file>, Page: <n>' so the text itself
    carries its origin into the embedding, BM25 and the LLM prompt.
    """
    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    chunks = []
    for page in pages:
        text = page["text"].strip()
        meta = page["metadata"]
        prefix = f"Source: {meta['source']}, Page: {meta['page']}\n\n"

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):  # avoid cutting a word in half
                cut = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
                if cut > start + overlap:
                    end = cut
            piece = text[start:end].strip()
            if piece:
                chunks.append({"text": prefix + piece, "metadata": dict(meta)})
            if end >= len(text):
                break
            start = end - overlap
            nxt = text.find(" ", start, end)  # start the next chunk at a word boundary
            if nxt != -1:
                start = nxt + 1
    return chunks