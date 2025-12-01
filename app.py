import os
import io
import json
import textwrap
import tempfile

from flask import Flask, request, send_file, render_template
from openai import OpenAI
import pdfplumber
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

# Uses OPENAI_API_KEY from your environment
client = OpenAI()
app = Flask(__name__)

client.timeout = 120.0


# ---------- BASIC PDF TEXT EXTRACTION ----------

def extract_pages(pdf_path):
    """Return a list of text strings, one per page."""
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
    return pages_text


# ---------- ASK THE MODEL WHAT TO FILL ----------

def get_fill_items_from_text(text):
    """
    Ask the model to find EVERY thing a student is supposed to fill in on this page:
    - Questions
    - Sentences with blanks
    - Bullet prompts that obviously want answers

    Returns a list like:
    [
      {"prompt": "...line or question exactly as on the page...", "answer": "...short answer..."},
      ...
    ]
    """
    if not text.strip():
        return []

    prompt = f"""
You are helping with a school worksheet.

You are given the FULL text of ONE worksheet page. This may include:
- Section titles and headers
- Instructions like "For questions 1–3, use Article 2, Clause 3"
- Labeled blanks like "Birth date: ______; Place of birth: ______"
- Numbered questions (1., 2., 3., etc.)
- Bullet lists, including bullets that are blank (for example "• ______")

Your job:

1. Find EVERY item a student is expected to fill in. This includes:
   - Direct questions ending in a question mark.
   - Prompts ending with a colon that clearly require an answer
     (for example, "The Supreme Court can hear cases that:").
   - Labeled blanks such as "Birth date: ______; Place of birth: ______".
   - Bullet prompts where the bullet points are where the student would write.

2. For EACH such item, output ONE object:

   {{
     "prompt": "the exact line / question / label from the page that the answer belongs to",
     "answer": "a short, direct answer the student would write"
   }}

   IMPORTANT RULES ABOUT THE ANSWER:
   - The answer MUST NOT contain blank placeholders like "______" or "___".
   - The answer MUST be actual content (names, dates, phrases, explanations).
   - The answer MUST NOT be identical to the prompt text.
   - If the prompt line already contains labels (e.g. "Birth date: ______; Place of birth: ______"),
     fill them with real information, e.g. "Birth date: 1798; Place of birth: Etables, France".
   - For bullet-style answers, you may use multiple short phrases separated by semicolons
     or put them on separate lines inside the answer string (using "\\n").

3. Use any relevant instructions or headings that appear ABOVE the prompt text as context.
   If there is no content on the page, you may use your own general knowledge.

4. Always give your best short answer. Do NOT say things like "not in text",
   "unavailable", or "cannot answer". Just answer based on your knowledge.

5. For bullet-style prompts, put each answer on its own line using "\n", e.g.:

"• case type one\n• case type two\n• case type three"

Return ONLY valid JSON in this exact format:

[
  {{
    "prompt": "…",
    "answer": "…"
  }},
  ...
]

Do not include any other text, no explanations.

Page text:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("Warning: JSON parse failed for one page, skipping its answers.")
        return []

    cleaned = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            prompt_text = (item.get("prompt") or "").strip()
            answer_text = (item.get("answer") or "").strip()

            if not prompt_text or not answer_text:
                continue

            # If the answer still looks like blanks or just copies the prompt, skip it.
            # 1) Mostly underscores?
            if "_" in answer_text:
                non_underscores = "".join(ch for ch in answer_text if ch != "_").strip()
                # if after removing underscores there's almost nothing left, ignore it
                if len(non_underscores) < max(3, len(answer_text) // 4):
                    continue

            # 2) Answer basically equals prompt?
            if answer_text.lower().strip(" .;:") == prompt_text.lower().strip(" .;:"):
                continue

            cleaned.append({"prompt": prompt_text, "answer": answer_text})

    return cleaned



def build_items_for_pdf(pdf_path):
    """For each page, get a list of fill items (prompt + answer)."""
    pages = extract_pages(pdf_path)
    all_items = []

    for i, page_text in enumerate(pages):
        print(f"Processing page {i + 1}/{len(pages)}...")
        items = get_fill_items_from_text(page_text)
        print(f"  Found {len(items)} fillable items.")
        all_items.append(items)

    return all_items


# ---------- FIND WHERE A PROMPT LIVES ON THE PAGE ----------

def find_prompt_anchor(pl_page, prompt_text):
    """
    Try to find where the prompt appears on the page using words + positions.

    Returns (x_from_left, y_from_bottom) in PDF coordinates for where to place
    the answer, or None if not found.

    This keeps the logic simple:
    - Take the first few words of the prompt.
    - Slide over all words on the page and score how many match in order.
    - Require at least a few matches, then use that snippet's position.
    """
    words = pl_page.extract_words()
    if not words:
        return None

    page_height = pl_page.height

    p_words = prompt_text.split()
    if not p_words:
        return None

    snippet_len = min(len(p_words), 6)
    snippet = [w.strip(".,?!:;").lower() for w in p_words[:snippet_len]]
    if not snippet:
        return None

    page_words = [w["text"] for w in words]

    best_index = None
    best_score = 0

    for i in range(0, len(page_words) - snippet_len + 1):
        window = page_words[i:i + snippet_len]
        window_clean = [w.strip(".,?!:;").lower() for w in window]
        score = sum(1 for a, b in zip(snippet, window_clean) if a == b)
        if score > best_score:
            best_score = score
            best_index = i

    # Require at least 3 matching words (or almost all)
    min_required = max(3, snippet_len - 1)
    if best_index is None or best_score < min_required:
        return None

    snippet_words = words[best_index:best_index + snippet_len]
    x0 = min(float(w["x0"]) for w in snippet_words)
    bottoms = [float(w["bottom"]) for w in snippet_words]
    bottom_from_top = max(bottoms)

    y_from_bottom = page_height - bottom_from_top
    return x0, y_from_bottom


# ---------- WRITE ANSWERS BACK TO THE PDF ----------

def overlay_answers_on_pdf(original_pdf, items_by_page, output_pdf):
    """
    Draw answers onto a copy of the original PDF.

    For each page:
    - For each (prompt, answer), find where the prompt's text is.
    - Draw the answer underneath that line, left-aligned with the prompt.
    - If we can't find the prompt, we simply skip that answer.
    """
    reader = PdfReader(original_pdf)
    writer = PdfWriter()
    num_pages = len(reader.pages)

    with pdfplumber.open(original_pdf) as pl_doc:
        for i in range(num_pages):
            page = reader.pages[i]

            if i >= len(items_by_page) or not items_by_page[i]:
                # No answers for this page; just copy
                writer.add_page(page)
                continue

            pl_page = pl_doc.pages[i]
            page_height = float(page.mediabox.height)
            page_width = float(page.mediabox.width)

            # Create a transparent overlay with reportlab
            packet = io.BytesIO()
            can = canvas.Canvas(packet, pagesize=(page_width, page_height))
            can.setFont("Helvetica", 9)

            for item in items_by_page[i]:
                prompt_text = item.get("prompt", "") or ""
                answer_text = item.get("answer", "") or ""
                if not prompt_text or not answer_text:
                    continue

                anchor = find_prompt_anchor(pl_page, prompt_text)
                if anchor is None:
                    # Can't find where this line is; skip to avoid random placement
                    continue

                base_x, base_y = anchor
                x = base_x
                y = base_y - 14  # a bit below the prompt line

                # Wrap the answer into lines so it doesn't run off the page
                wrap_width = 80  # number of characters per line (rough)
                lines = []
                for chunk in answer_text.split("\n"):
                    lines.extend(textwrap.wrap(chunk, width=wrap_width) or [""])

                for line in lines:
                    y -= 12
                    if y < 40:  # don't draw into the bottom margin
                        break
                    can.drawString(x, y, line)

            can.save()
            packet.seek(0)

            try:
                overlay_pdf = PdfReader(packet)
                if len(overlay_pdf.pages) == 0:
                    # Nothing actually drawn; just copy the page
                    writer.add_page(page)
                    continue

                overlay_page = overlay_pdf.pages[0]
                page.merge_page(overlay_page)
            except Exception as e:
                # If anything goes wrong with merging, log and fall back to original page
                print(f"Warning: overlay merge failed on page {i+1}: {e}")
                # Just keep original page

            writer.add_page(page)

    with open(output_pdf, "wb") as f:
        writer.write(f)


# ---------- FLASK ROUTES ----------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", error=None)

    if "pdf_file" not in request.files:
        return render_template("index.html", error="No file part in request.")

    file = request.files["pdf_file"]
    if file.filename == "":
        return render_template("index.html", error="No file selected.")

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_in:
        input_path = tmp_in.name
        file.save(input_path)

    # Prepare temp output path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_out:
        output_path = tmp_out.name

    try:
        print("Building fill items for uploaded PDF...")
        items_by_page = build_items_for_pdf(input_path)

        print("Overlaying answers onto PDF...")
        overlay_answers_on_pdf(input_path, items_by_page, output_path)

        return send_file(
            output_path,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="worksheet_filled.pdf"
        )
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        # output_path can be left; the container/WSL session will clean up eventually


if __name__ == "__main__":
    # For local testing: http://localhost:5000
    app.run(host="0.0.0.0", port=5000, debug=True)
