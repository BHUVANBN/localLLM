from flask import Flask, request, render_template_string, redirect, url_for
from interface.chat import generate
from rag.faiss_index import build_index, query_index
import os
import tempfile

app = Flask(__name__)

# Global in-memory index
INDEX = None
CHUNKS = None
CURRENT_PDF = "data/pdfs/sample.pdf"

def ensure_index(pdf_path: str):
    global INDEX, CHUNKS, CURRENT_PDF
    if INDEX is None or CHUNKS is None or pdf_path != CURRENT_PDF:
        INDEX, CHUNKS = build_index(pdf_path)
        CURRENT_PDF = pdf_path


PAGE = '''
<h2>TinyLLM Chat with PDF</h2>
<form method="POST" action="/upload" enctype="multipart/form-data" style="margin-bottom: 1em;">
  <input type="file" name="pdf" accept="application/pdf">
  <input type="submit" value="Upload PDF">
  {% if current_pdf %}<span style="margin-left:1em;">Current PDF: {{ current_pdf }}</span>{% endif %}
  <div style="font-size: 0.9em; color: #555;">If none uploaded, using default: data/pdfs/sample.pdf</div>
  <hr>
</form>

<form method="POST">
  <label>Question:</label>
  <input name="query" style="width:500px;" autocomplete="off">
  <input type="submit" value="Ask">
</form>

{% if relevant %}
<h4>Retrieved context:</h4>
<ul>
  {% for c in relevant %}<li>{{ c }}</li>{% endfor %}
  </ul>
{% endif %}

{% if answer %}
<h3>Answer:</h3>
<div style="white-space: pre-wrap; border: 1px solid #ddd; padding: 10px;">{{answer}}</div>
{% endif %}
'''


@app.route("/", methods=["GET", "POST"])
def chat():
    ensure_index(CURRENT_PDF)
    answer = None
    relevant = None
    if request.method == "POST":
        query = request.form.get("query", "")
        if query:
            relevant = query_index(INDEX, CHUNKS, query, top_k=3)
            prompt = "\n".join(relevant) + f"\n\nQuestion: {query}\nAnswer:"
            answer = generate(prompt)
    return render_template_string(PAGE, answer=answer, relevant=relevant, current_pdf=CURRENT_PDF)


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("pdf")
    if not file or file.filename == "":
        return redirect(url_for("chat"))
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    file.save(path)
    # build index for uploaded file
    ensure_index(path)
    return redirect(url_for("chat"))


if __name__ == "__main__":
    ensure_index(CURRENT_PDF)
    app.run(debug=True)
