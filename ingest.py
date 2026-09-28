import sys
import os
import config
from document_processor import extract_text_and_metadata_from_pdf, chunk_text_with_context
from vector_db import VectorDB
from llm import generate_hypothetical_questions

def main(pdf_path: str):
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    print(f"Processing {pdf_path}...")
    pages = extract_text_and_metadata_from_pdf(pdf_path)
    
    print("Chunking text with context...")
    chunks = chunk_text_with_context(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    
    print(f"Generated {len(chunks)} chunks. Generating Reverse-HyDE questions...")
    for i, chunk in enumerate(chunks):
        questions = generate_hypothetical_questions(chunk["text"])
        chunk["hypothetical_questions"] = questions
        print(f"Processed chunk {i+1}/{len(chunks)}")
        
    print("Adding to Vector DB & BM25...")
    db = VectorDB()
    doc_id = os.path.basename(pdf_path)
    db.add_chunks(chunks, doc_id)
    
    print("Ingestion complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_pdf>")
    else:
        main(sys.argv[1])
