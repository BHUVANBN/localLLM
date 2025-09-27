from flask import Flask, request, render_template_string
from rag.faiss_index import build_index, query_index
from model.train_llm import TinyLLM
import torch
import sentencepiece as spm

app = Flask(__name__)

# Load tokenizer & model
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/mytok.model")
vocab_size = sp.get_piece_size()
device = torch.device("cpu")
model = TinyLLM(vocab_size).to(device)
model.load_state_dict(torch.load("model/tiny_llm.pt", map_location=device))
model.eval()

# FAISS index
index, chunks = build_index("data/pdfs/sample.pdf")

# Simple page
HTML = '''
<form method="POST">
  Question: <input name="query" style="width:400px;">
  <input type="submit" value="Ask">
</form>
{% if answer %}
<h3>Answer:</h3>
<p>{{answer}}</p>
{% endif %}
'''

def generate(prompt, max_new_tokens=50):
    ids = sp.encode(prompt, out_type=int)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)
            next_id = torch.argmax(logits[0, -1]).unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_id], dim=1)
    return sp.decode(input_ids[0].tolist())

@app.route("/", methods=["GET", "POST"])
def chat():
    answer = None
    if request.method == "POST":
        query = request.form["query"]
        relevant = query_index(index, chunks, query, top_k=2)
        prompt = "\n".join(relevant) + f"\n\nQuestion: {query}\nAnswer:"
        answer = generate(prompt)
    return render_template_string(HTML, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
