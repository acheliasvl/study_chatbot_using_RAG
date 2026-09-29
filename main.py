"""Terminal RAG chatbot over PDFs.

  python main.py                     start chat
  python main.py notes.pdf book.pdf  index these PDFs first, then chat

In chat:  /add <path>   /docs   /verbose   /help   /quit
"""
import argparse
import os
import sys

import config
from document_processor import chunk_pages, extract_pages
from llm import check_ollama, generate_answer_stream, generate_hypothetical_questions
from vector_db import VectorDB

HELP = """\
  /add <path>   index a PDF (or every PDF in a folder); re-adding replaces it
  /docs         list indexed documents
  /verbose      show retrieval stages (dense, BM25, RRF, rerank) for each question
  /quit         exit
  anything else = a question"""


def clean_path(raw: str) -> str:
    """Drag-and-drop / copy-paste paths often come wrapped in quotes."""
    return os.path.abspath(os.path.expanduser(raw.strip().strip("\"'")))


# ── Phase 1: indexing ─────────────────────────────────────────────────────────
def ingest_pdf(db: VectorDB, path: str):
    if not path.lower().endswith(".pdf"):
        print(f"  skip (not a PDF): {path}")
        return

    print(f"  extracting {os.path.basename(path)} ...")
    pages = extract_pages(path)
    if not pages:
        print("  no text found (scanned PDF? needs OCR)")
        return

    chunks = chunk_pages(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"  {len(pages)} pages -> {len(chunks)} chunks")

    if config.USE_REVERSE_HYDE:
        for i, chunk in enumerate(chunks, 1):
            chunk["questions"] = generate_hypothetical_questions(chunk["text"], config.HYDE_NUM_QUESTIONS)
            print(f"\r  reverse-HyDE questions {i}/{len(chunks)}", end="", flush=True)
        print()

    def progress(done, total):
        print(f"\r  embedding {done}/{total}", end="", flush=True)

    if db.has_document(path):
        print("  already indexed -> replacing old version")
        db.delete_document(path)
    db.add_chunks(chunks, path, on_progress=progress)
    print(f"\n  done: {os.path.basename(path)}")


def ingest(db: VectorDB, raw_path: str):
    path = clean_path(raw_path)
    if os.path.isdir(path):
        pdfs = sorted(os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".pdf"))
        if not pdfs:
            print(f"  no PDFs in {path}")
        for pdf in pdfs:
            ingest_pdf(db, pdf)
    elif os.path.isfile(path):
        ingest_pdf(db, path)
    else:
        print(f"  file not found: {path}")


# ── Phase 2: retrieval + generation ───────────────────────────────────────────
def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(c["text"] for c in chunks)  # each chunk starts with Source/Page
    return (
        "You are a study assistant. Answer the question using ONLY the context below.\n"
        "Cite sources as (file, page). If the context does not contain the answer, say so.\n"
        "Answer in the language of the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def print_trace(trace: dict):
    for stage in ("dense", "bm25", "rrf", "rerank"):
        print(f"  [{stage}]")
        for label, score in trace.get(stage, []):
            print(f"     {label}" + (f"   {score:.4f}" if score is not None else ""))


def answer(db: VectorDB, query: str, verbose: bool):
    chunks = db.search(query)
    if verbose:
        print_trace(db.last_trace)
    if not chunks:
        print("No relevant context found.")
        return

    print("\nAssistant: ", end="", flush=True)
    try:
        for token in generate_answer_stream(build_prompt(query, chunks)):
            print(token, end="", flush=True)
    except KeyboardInterrupt:
        print("\n  [stopped]")
    print()

    seen = []
    for c in chunks:
        ref = f"{c['metadata']['source']} p.{c['metadata']['page']}"
        if ref not in seen:
            seen.append(ref)
    print("Sources: " + ", ".join(seen))


def main():
    parser = argparse.ArgumentParser(description="Terminal RAG chatbot over PDFs")
    parser.add_argument("paths", nargs="*", help="PDF files or folders to index before chatting")
    parser.add_argument("-v", "--verbose", action="store_true", help="show retrieval stages")
    args = parser.parse_args()

    problems = check_ollama()
    if problems:
        print("\n".join(problems))
        sys.exit(1)

    db = VectorDB()
    for p in args.paths:
        ingest(db, p)

    verbose = args.verbose
    print(f"\nReady. {db.count()} chunks indexed. /help for commands.")
    if db.count() == 0:
        print("No documents yet. Use: /add <path to pdf>")

    while True:
        try:
            line = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        try:
            if line in ("/quit", "/exit", "exit", "quit"):
                break
            elif line == "/help":
                print(HELP)
            elif line == "/verbose":
                verbose = not verbose
                print(f"  verbose {'on' if verbose else 'off'}")
            elif line == "/docs":
                docs = db.list_documents()
                for d in docs:
                    print(f"  {d['source']}  ({d['chunks']} chunks)  {d['path']}")
                if not docs:
                    print("  (none)")
            elif line.startswith("/add"):
                arg = line[4:].strip()
                if arg:
                    ingest(db, arg)
                else:
                    print("  usage: /add <path>")
            elif line.startswith("/"):
                print("  unknown command, /help")
            elif db.count() == 0:
                print("  No documents yet. Use: /add <path to pdf>")
            else:
                answer(db, line, verbose)
        except KeyboardInterrupt:
            print("\n  [cancelled]")
        except Exception as e:  # keep the chat alive on Ollama/IO errors
            print(f"  error: {e}")


if __name__ == "__main__":
    main()