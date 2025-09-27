import torch
import sentencepiece as spm
from model.train_llm import TinyLLM
from rag.faiss_index import build_index, query_index

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/mytok.model")
vocab_size = sp.get_piece_size()

# Load tiny LLM
device = torch.device("cpu")
model = TinyLLM(vocab_size).to(device)
model.load_state_dict(torch.load("model/tiny_llm.pt", map_location=device))
model.eval()

# Build FAISS index for a PDF
index, chunks = build_index("data/pdfs/sample.pdf")

def generate(prompt, max_new_tokens=50):
    ids = sp.encode(prompt, out_type=int)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)
            next_id = torch.argmax(logits[0, -1]).unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_id], dim=1)
    return sp.decode(input_ids[0].tolist())

# Chat CLI
while True:
    query = input("\nYou: ")
    if query.lower() in ["exit", "quit"]:
        break
    relevant = query_index(index, chunks, query, top_k=2)
    prompt = "\n".join(relevant) + f"\n\nQuestion: {query}\nAnswer:"
    response = generate(prompt)
    print("LLM:", response)
