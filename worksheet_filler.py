import os
import sys
import json
import io
import textwrap

from openai import OpenAI
import pdfplumber
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

client = OpenAI()


def extract_pages(pdf_path):
    """Return a list of text strings, one per page."""
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
    return pages_text


def get_qna_from_text(text):
    """
    Ask the model to find questions in the text and answer them.
    Returns a Python list of {question, answer} dicts.
    """
    if not text.strip():
        return []

    prompt = f"""
You are helping with a school worksheet.

From the text below, find any questions a student is supposed to answer
and give short, direct answers.

Return ONLY valid JSON in this exact format:

[
  {{
    "question": "…question text exactly as it appears…",
    "answer": "…short answer only…"
  }},
  ...
]

Do not include any other text, no explanations.

Text:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            cleaned = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                q = item.get("question", "").strip()
                a = item.get("answer", "").strip()
                if q and a:
                    cleaned.append({"question": q, "answer": a})
            return cleaned
        else:
            return []
    except json.JSONDecodeError:
        print("Warning: JSON parse failed for one page, skipping its answers.")
        return []


def build_qna_for_pdf(pdf_path):
    """
    For each page, get a list of Q&A dicts.
    Returns a list: index = page number, value = list of Q&A for that page.
    """
    pages = extract_pages(pdf_path)
    all_qna = []

    for i, page_text in enumerate(pages):
        print(f"Processing page {i + 1}/{len(pages)}...")
        qna_list = get_qna_from_text(page_text)
        print(f"  Found {len(qna_list)} questions.")
        all_qna.append(qna_list)

    return all_qna


def find_question_anchor(pl_page, question):
    """
    Try to find where the question appears on the page using words + positions.
    Returns (x, y_from_bottom) in PDF coordinates for where to place the answer,
    or None if not found.

    pl_page: pdfplumber page
    question: string
    """
    words = pl_page.extract_words()
    if not words:
        return None

    page_height = pl_page.height

    # Split question into words, take a short snippet
    q_words = question.split()
    if not q_words:
        return None

    snippet_len = min(len(q_words), 5)
    snippet = [w.strip(".,?!:;").lower() for w in q_words[:snippet_len]]

    if not snippet:
        return None

    # Precompute page words text
    page_words = [w["text"] for w in words]

    best_index = None
    best_score = 0

    # Sliding window over page words
    for i in range(0, len(page_words) - snippet_len + 1):
        window = page_words[i:i + snippet_len]
        window_clean = [w.strip(".,?!:;").lower() for w in window]

        # Score = number of matching words in order
        score = sum(1 for a, b in zip(snippet, window_clean) if a == b)

        if score > best_score:
            best_score = score
            best_index = i

    # Require at least 2 matching words to avoid random matches
    if best_index is None or best_score < 2:
        return None

    # Use the bottom of the last word in that matched snippet
    anchor_word = words[best_index + snippet_len - 1]
    x0 = float(anchor_word["x0"])
    bottom_from_top = float(anchor_word["bottom"])

    # pdfplumber's origin is top-left; reportlab's origin is bottom-left.
    y_from_bottom = page_height - bottom_from_top

    return x0, y_from_bottom


def overlay_answers_on_pdf(original_pdf, qna_by_page, output_pdf):
    """
    Take the original PDF and write AI answers onto each page
    near each question if we can find it. Saves the result as output_pdf.
    """
    reader = PdfReader(original_pdf)
    writer = PdfWriter()
    num_pages = len(reader.pages)

    # Open with pdfplumber for layout info
    with pdfplumber.open(original_pdf) as pl_doc:
        for i in range(num_pages):
            page = reader.pages[i]

            if i >= len(qna_by_page) or not qna_by_page[i]:
                # No Q&A for this page, just copy
                writer.add_page(page)
                continue

            pl_page = pl_doc.pages[i]
            page_height = float(page.mediabox.height)
            page_width = float(page.mediabox.width)

            # Create overlay
            packet = io.BytesIO()
            can = canvas.Canvas(packet, pagesize=(page_width, page_height))
            can.setFont("Helvetica", 9)

            # Fallback writing position if we can't find question
            fallback_margin_x = 50
            fallback_y = 150
            fallback_wrap_width = 90

            for qa in qna_by_page[i]:
                question = qa.get("question", "")
                answer = qa.get("answer", "")

                if not answer:
                    continue

                # Try to find anchor location for this question
                anchor = find_question_anchor(pl_page, question)
                if anchor is not None:
                    base_x, base_y = anchor
                    # Start a bit below the question
                    x = base_x
                    y = base_y - 15
                    wrap_width = 80  # fewer chars because we're near the margin
                else:
                    # Fallback: dump near bottom of page
                    x = fallback_margin_x
                    y = fallback_y
                    wrap_width = fallback_wrap_width
                    fallback_y -= 60  # next fallback answer lower

                # Only draw the ANSWER (not the question)
                a_text = answer
                a_lines = textwrap.wrap(a_text, width=wrap_width)

                for line in a_lines:
                    y -= 12
                    if y < 40:
                        break
                    can.drawString(x, y, line)

            can.save()
            packet.seek(0)

            overlay_pdf = PdfReader(packet)
            overlay_page = overlay_pdf.pages[0]

            page.merge_page(overlay_page)
            writer.add_page(page)

    with open(output_pdf, "wb") as f:
        writer.write(f)


def main():
    if len(sys.argv) != 3:
        print("Usage: python worksheet_filler.py input.pdf output.pdf")
        sys.exit(1)

    input_pdf = sys.argv[1]
    output_pdf = sys.argv[2]

    if not os.path.exists(input_pdf):
        print(f"File not found: {input_pdf}")
        sys.exit(1)

    print("Building Q&A for PDF...")
    qna_by_page = build_qna_for_pdf(input_pdf)

    print("Writing answers back into PDF...")
    overlay_answers_on_pdf(input_pdf, qna_by_page, output_pdf)

    print(f"Done. Saved filled PDF to: {output_pdf}")


if __name__ == "__main__":
    main()
