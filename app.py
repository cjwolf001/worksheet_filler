import os
import io
import json
import textwrap
import tempfile

from flask import Flask, request, send_file, render_template_string
from openai import OpenAI
import pdfplumber
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

# Uses OPENAI_API_KEY from your environment
client = OpenAI()
app = Flask(__name__)


# --------- CORE WORKSHEET LOGIC ---------

def extract_pages(pdf_path):
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
    return pages_text


def get_qna_from_text(text):
    if not text.strip():
        return []

    prompt = f"""
You are helping with a school worksheet.

You are given the FULL text of a worksheet page. This may include:
- Section titles and headers
- Instructions like "For questions 1–3, use Article 2, Clause 3"
- Numbered questions (1., 2., 3., etc.)
- Answer blanks or lines

Your job:
1. Find EVERY actual question a student is supposed to answer.
2. Use any relevant instructions or headings that appear ABOVE the question
   (for example: "For questions 1–3, refer to Article 2, Clause 3").
3. Give a short, direct answer to each question.
4. If a question cannot be answered from the page text alone, you MAY use
   your own general knowledge or web search to answer it.

IMPORTANT:
- Do NOT treat pure instructions or headings as questions.
  Example: "For questions 1–3, use the chart above." is NOT a question.
- Questions may be numbered (1, 2, 3) or written in sentences.
- Always give your best short answer for every question you find.
- Do NOT say things like "not in text", "unavailable", or "cannot answer".

Return ONLY valid JSON in this exact format:

[
  {{
    "question": "…the question text as it appears on the page…",
    "answer": "…short answer only…"
  }},
  ...
]

Do not include any other text, no explanations.

Page text:
{text}
"""

    # Use the Responses API with web_search enabled
    response = client.responses.create(
        model="gpt-5.1",  # must be a model that supports web_search
        input=prompt,
        tools=[
            {"type": "web_search"}
        ]
    )

    # Responses API: take the main text output
    output = response.output[0].content[0].text

    try:
        data = json.loads(output)
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
    pages = extract_pages(pdf_path)
    all_qna = []

    for i, page_text in enumerate(pages):
        print(f"Processing page {i + 1}/{len(pages)}...")
        qna_list = get_qna_from_text(page_text)
        print(f"  Found {len(qna_list)} questions.")
        all_qna.append(qna_list)

    return all_qna


def find_question_anchor(pl_page, question):
    words = pl_page.extract_words()
    if not words:
        return None

    page_height = pl_page.height

    q_words = question.split()
    if not q_words:
        return None

    snippet_len = min(len(q_words), 5)
    snippet = [w.strip(".,?!:;").lower() for w in q_words[:snippet_len]]

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

    if best_index is None or best_score < 2:
        return None

    anchor_word = words[best_index + snippet_len - 1]
    x0 = float(anchor_word["x0"])
    bottom_from_top = float(anchor_word["bottom"])

    y_from_bottom = page_height - bottom_from_top
    return x0, y_from_bottom


def overlay_answers_on_pdf(original_pdf, qna_by_page, output_pdf):
    reader = PdfReader(original_pdf)
    writer = PdfWriter()
    num_pages = len(reader.pages)

    with pdfplumber.open(original_pdf) as pl_doc:
        for i in range(num_pages):
            page = reader.pages[i]

            if i >= len(qna_by_page) or not qna_by_page[i]:
                writer.add_page(page)
                continue

            pl_page = pl_doc.pages[i]
            page_height = float(page.mediabox.height)
            page_width = float(page.mediabox.width)

            packet = io.BytesIO()
            can = canvas.Canvas(packet, pagesize=(page_width, page_height))
            can.setFont("Helvetica", 9)

            fallback_margin_x = 50
            fallback_y = 150
            fallback_wrap_width = 90

            for qa in qna_by_page[i]:
                question = qa.get("question", "")
                answer = qa.get("answer", "")

                if not answer:
                    continue

                anchor = find_question_anchor(pl_page, question)
                if anchor is not None:
                    base_x, base_y = anchor
                    x = base_x
                    y = base_y - 15
                    wrap_width = 80
                else:
                    x = fallback_margin_x
                    y = fallback_y
                    wrap_width = fallback_wrap_width
                    fallback_y -= 60

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


# ----------------- SIMPLE HTML FRONTEND -----------------

INDEX_HTML = """
<!doctype html>
<html>
<head>
    <title>Worksheet Filler</title>
</head>
<body>
    <h1>Conor's AI Worksheet Filler</h1>
    <p>Upload a PDF worksheet. The AI will try to answer and fill it.</p>
    <p> I am personally paying for these API tokens, please don't abuse!" </p>
    <form method="post" action="/" enctype="multipart/form-data">
        <input type="file" name="pdf_file" accept="application/pdf" required>
        <br><br>
        <button type="submit">Upload & Fill</button>
    </form>
    {% if error %}
      <p style="color:red;">{{ error }}</p>
    {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(INDEX_HTML, error=None)

    if "pdf_file" not in request.files:
        return render_template_string(INDEX_HTML, error="No file part in request.")

    file = request.files["pdf_file"]
    if file.filename == "":
        return render_template_string(INDEX_HTML, error="No file selected.")

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_in:
        input_path = tmp_in.name
        file.save(input_path)

    # Prepare temp output path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_out:
        output_path = tmp_out.name

    try:
        print("Building Q&A for uploaded PDF...")
        qna_by_page = build_qna_for_pdf(input_path)

        print("Overlaying answers...")
        overlay_answers_on_pdf(input_path, qna_by_page, output_path)

        # Send the result back
        return send_file(
            output_path,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="worksheet_filled.pdf"
        )
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        # you can also remove output_path later if you want, but send_file needs it alive


if __name__ == "__main__":
    # For WSL or Render local testing, hit http://localhost:5000 in your browser
    app.run(host="0.0.0.0", port=5000, debug=True)
