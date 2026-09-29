import os
import fitz


def extract_pages(pdf_path: str) -> list[dict]:
    """
    PDF -> pages containing structured text blocks.

    Each block keeps:
      - text
      - font size
      - page number
      - source

    Font size is later used to detect likely headings.
    """
    source = os.path.basename(pdf_path)
    pages = []

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):

            blocks = []

            # "dict" gives us spans with font information
            page_dict = page.get_text("dict")

            for block in page_dict["blocks"]:
                if block["type"] != 0:  # ignore images/drawings
                    continue

                block_text = []

                for line in block["lines"]:
                    line_text = []

                    for span in line["spans"]:
                        text = span["text"].strip()

                        if text:
                            line_text.append({
                                "text": text,
                                "size": span["size"],
                                "bold": bool(span["flags"] & 16)
                            })

                    if line_text:
                        block_text.append(line_text)

                if block_text:
                    blocks.append(block_text)

            if blocks:
                pages.append({
                    "blocks": blocks,
                    "metadata": {
                        "source": source,
                        "page": page_num
                    }
                })

    return pages


def chunk_by_topics(
    pages: list[dict],
    chunk_size: int,
    overlap: int
) -> list[dict]:
    """
    Create topic-aware chunks.

    Headings are detected using:
      - larger-than-normal font size
      - bold text
      - common heading patterns such as:
          1.
          1.1
          1.1.1
          Chapter 1
          Section 2
          etc.

    Text belonging to the same topic stays together.

    If a topic is larger than chunk_size, it is split into
    smaller overlapping chunks.
    """

    if overlap >= chunk_size:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE"
        )

    # ---------------------------------------------------------
    # STEP 1 — Flatten blocks and calculate normal font size
    # ---------------------------------------------------------

    all_blocks = []

    for page in pages:
        for block in page["blocks"]:

            lines = []

            for line in block:
                text = " ".join(span["text"] for span in line).strip()

                if not text:
                    continue

                sizes = [span["size"] for span in line]
                bold = any(span["bold"] for span in line)

                lines.append({
                    "text": text,
                    "size": max(sizes),
                    "bold": bold
                })

            if lines:
                all_blocks.append({
                    "lines": lines,
                    "metadata": page["metadata"]
                })

    if not all_blocks:
        return []

    # Normal body-text size
    all_sizes = []

    for block in all_blocks:
        for line in block["lines"]:
            all_sizes.append(line["size"])

    all_sizes.sort()

    median_size = all_sizes[len(all_sizes) // 2]

    # ---------------------------------------------------------
    # STEP 2 — Detect headings
    # ---------------------------------------------------------

    import re

    def looks_like_heading(line: dict) -> bool:
        text = line["text"].strip()
        size = line["size"]
        bold = line["bold"]

        if not text:
            return False

        # Avoid treating huge paragraphs as headings
        if len(text) > 150:
            return False

        # Common numbered headings:
        # 1 Introduction
        # 1.1 Processes
        # 1.1.1 Threads
        numbered = re.match(
            r"^\d+(\.\d+)*[\.)]?\s+\S+",
            text
        )

        # Chapter 1, Chapter One, etc.
        chapter = re.match(
            r"^(chapter|section)\s+\w+",
            text,
            re.IGNORECASE
        )

        # A significantly larger font is a strong signal
        large_font = size >= median_size * 1.20

        # Short bold lines are often headings
        bold_heading = bold and len(text.split()) <= 15

        return bool(
            numbered
            or chapter
            or large_font
            or bold_heading
        )

    # ---------------------------------------------------------
    # STEP 3 — Build topic sections
    # ---------------------------------------------------------

    topics = []

    current_topic = None
    current_text = []

    current_metadata = None

    def save_topic():
        nonlocal current_topic, current_text, current_metadata

        if not current_text:
            return

        text = "\n".join(current_text).strip()

        if not text:
            return

        topics.append({
            "topic": current_topic,
            "text": text,
            "metadata": dict(current_metadata)
        })

    for block in all_blocks:

        page_metadata = block["metadata"]

        for line in block["lines"]:

            text = line["text"]

            if looks_like_heading(line):

                # Save previous topic
                save_topic()

                # Start new topic
                current_topic = text
                current_text = []

                current_metadata = {
                    **page_metadata,
                    "section": current_topic
                }

            else:

                # If the document starts with text before
                # the first detected heading
                if current_metadata is None:
                    current_metadata = {
                        **page_metadata,
                        "section": "Introduction"
                    }

                current_text.append(text)

    # Save final topic
    save_topic()

    # ---------------------------------------------------------
    # STEP 4 — Split large topics into chunks
    # ---------------------------------------------------------

    chunks = []

    for topic in topics:

        text = topic["text"]
        metadata = topic["metadata"]
        section = topic["topic"] or metadata.get(
            "section",
            "Unknown"
        )

        prefix = (
            f"Source: {metadata['source']}\n"
            f"Page: {metadata['page']}\n"
            f"Section: {section}\n\n"
        )

        # If topic fits inside one chunk
        if len(text) <= chunk_size:

            chunks.append({
                "text": prefix + text,
                "metadata": dict(metadata)
            })

            continue

        # -----------------------------------------------------
        # Large topic -> overlapping chunks
        # -----------------------------------------------------

        start = 0

        while start < len(text):

            end = min(start + chunk_size, len(text))

            if end < len(text):

                # Prefer paragraph/line boundary
                cut = max(
                    text.rfind("\n", start, end),
                    text.rfind(" ", start, end)
                )

                if cut > start + overlap:
                    end = cut

            piece = text[start:end].strip()

            if piece:

                chunks.append({
                    "text": prefix + piece,
                    "metadata": dict(metadata)
                })

            if end >= len(text):
                break

            start = end - overlap

            # Move to next word boundary
            next_space = text.find(" ", start, end)

            if next_space != -1:
                start = next_space + 1

    # ---------------------------------------------------------
    # STEP 5 — Add chunk IDs
    # ---------------------------------------------------------

    for i, chunk in enumerate(chunks):

        chunk["metadata"]["chunk_id"] = i

    return chunks