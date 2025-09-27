# cli.py
from interface.chat import generate, query_index, build_index
import argparse

parser = argparse.ArgumentParser(description="TinyLLM CLI with RAG over a PDF")
parser.add_argument("--pdf", type=str, default="data/pdfs/sample.pdf", help="Path to a PDF to chat over")
args = parser.parse_args()

# Build FAISS index for chosen PDF
index, chunks = build_index(args.pdf)

print("=== TinyLLM CLI ===")
print("Type 'exit' or 'quit' to stop\n")

while True:
    query = input("You: ").strip()
    if query.lower() in ["exit", "quit"]:
        break
    relevant = query_index(index, chunks, query, top_k=2)
    prompt = "\n".join(relevant) + f"\n\nQuestion: {query}\nAnswer:"
    response = generate(prompt)
    print("LLM:", response)
