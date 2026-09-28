import sys
import os
import config
from document_processor import extract_text_from_pdf, chunk_text
from vector_db import VectorDB

def main(pdf_path: str):
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    print(f"Processing {pdf_path}...")
    text = extract_text_from_pdf(pdf_path)
    
    print("Chunking text...")
    chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    
    print(f"Generated {len(chunks)} chunks. Adding to Vector DB...")
    db = VectorDB()
    doc_id = os.path.basename(pdf_path)
    db.add_chunks(chunks, doc_id)
    
    print("Ingestion complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_pdf>")
    else:
        main(sys.argv[1])
