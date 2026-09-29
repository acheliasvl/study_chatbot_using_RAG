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
    Create topic-aware chunks from structured PDF text.

    Each chunk contains:
        - chunk_id
        - title
        - section
        - context
        - source
        - page

    Headings are detected using:
        - numbered heading patterns
        - Chapter / Section patterns
        - larger font size
        - bold text

    Large topics are split into smaller overlapping chunks.
    """

    if overlap >= chunk_size:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE"
        )

    import re

    # ---------------------------------------------------------
    # STEP 1 — Flatten PDF blocks
    # ---------------------------------------------------------

    all_blocks = []

    for page in pages:

        for block in page["blocks"]:

            lines = []

            for line in block:

                text = " ".join(
                    span["text"] for span in line
                ).strip()

                if not text:
                    continue

                sizes = [
                    span["size"]
                    for span in line
                ]

                bold = any(
                    span["bold"]
                    for span in line
                )

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

    # ---------------------------------------------------------
    # STEP 2 — Calculate normal body font size
    # ---------------------------------------------------------

    all_sizes = []

    for block in all_blocks:
        for line in block["lines"]:
            all_sizes.append(line["size"])

    all_sizes.sort()

    median_size = all_sizes[len(all_sizes) // 2]

    # ---------------------------------------------------------
    # STEP 3 — Heading detection
    # ---------------------------------------------------------

    def heading_level(text: str) -> int | None:

        # Ignore bullets
        if text in {"•", "-", "–", "—", "*"}:
            return None

        # Ignore very long lines
        if len(text) > 150:
            return None

        # Chapter 1 / Chapter One
        if re.match(
            r"^chapter\s+\w+",
            text,
            re.IGNORECASE
        ):
            return 1

        # Section 1 / Section 1.2
        if re.match(
            r"^section\s+\d+(\.\d+)*",
            text,
            re.IGNORECASE
        ):
            return 2

        # 1 Introduction
        # 1.1 Process Management
        # 1.1.1 Threads
        match = re.match(
            r"^(\d+(?:\.\d+)*)[\.)]?\s+\S+",
            text
        )

        if match:
            number = match.group(1)
            return number.count(".") + 1

        return None

    def looks_like_heading(line: dict) -> bool:

        text = line["text"].strip()
        size = line["size"]
        bold = line["bold"]

        if not text:
            return False

        # Never treat bullets as headings
        if text in {"•", "-", "–", "—", "*"}:
            return False

        # Explicit heading patterns
        if heading_level(text) is not None:
            return True

        # Larger font + reasonably short text
        large_font = (
            size >= median_size * 1.20
            and len(text.split()) <= 15
        )

        # Bold + short text
        bold_heading = (
            bold
            and len(text.split()) <= 12
        )

        return large_font or bold_heading

    # ---------------------------------------------------------
    # STEP 4 — Build topics
    # ---------------------------------------------------------

    topics = []

    current_title = "Introduction"
    current_section = "Introduction"
    current_text = []
    current_metadata = None

    def save_topic():

        if not current_text:
            return

        text = "\n".join(
            current_text
        ).strip()

        if not text:
            return

        topics.append({
            "title": current_title,
            "section": current_section,
            "text": text,
            "metadata": dict(current_metadata)
        })

    for block in all_blocks:

        page_metadata = block["metadata"]

        for line in block["lines"]:

            text = line["text"].strip()

            if looks_like_heading(line):

                # Save previous topic
                save_topic()

                # Determine heading level
                level = heading_level(text)

                # If it is an explicit numbered heading,
                # use it as the new section/title.
                if level == 1:

                    current_section = text
                    current_title = text

                elif level == 2:

                    current_title = text

                elif level and level >= 3:

                    current_title = text

                else:

                    # Font-based heading
                    current_title = text

                current_text = []

                current_metadata = {
                    **page_metadata
                }

            else:

                if current_metadata is None:
                    current_metadata = {
                        **page_metadata
                    }

                current_text.append(text)

    # Save final topic
    save_topic()

    # ---------------------------------------------------------
    # STEP 5 — Create chunks from topics
    # ---------------------------------------------------------

    chunks = []

    chunk_id = 0

    for topic in topics:

        text = topic["text"]

        title = topic["title"]
        section = topic["section"]

        metadata = topic["metadata"]

        # Context used for retrieval / embedding
        context = (
            f"This content is from the section "
            f"'{section}', specifically the topic "
            f"'{title}'."
        )

        prefix = (
            f"Source: {metadata['source']}\n"
            f"Page: {metadata['page']}\n"
            f"Section: {section}\n"
            f"Title: {title}\n"
            f"Context: {context}\n\n"
        )

        # -----------------------------------------------------
        # Topic fits into one chunk
        # -----------------------------------------------------

        if len(text) <= chunk_size:

            chunks.append({
                "text": prefix + text,

                "metadata": {
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "section": section,
                    "title": title,
                    "context": context,
                    "chunk_id": chunk_id
                }
            })

            chunk_id += 1
            continue

        # -----------------------------------------------------
        # Topic is too large → split it
        # -----------------------------------------------------

        start = 0

        while start < len(text):

            end = min(
                start + chunk_size,
                len(text)
            )

            if end < len(text):

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

                    "metadata": {
                        "source": metadata["source"],
                        "page": metadata["page"],
                        "section": section,
                        "title": title,
                        "context": context,
                        "chunk_id": chunk_id
                    }
                })

                chunk_id += 1

            if end >= len(text):
                break

            start = end - overlap

            next_space = text.find(
                " ",
                start,
                end
            )

            if next_space != -1:
                start = next_space + 1

    return chunks