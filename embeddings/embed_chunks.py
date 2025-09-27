import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def embed_text(text_list):
    embeddings = []
    for text in text_list:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, return_dict=True)
            emb = out.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(emb)
    return np.array(embeddings)

if __name__ == "__main__":
    from rag.pdf_chunks import extract_chunks
    chunks = extract_chunks("data/pdfs/sample.pdf")
    vectors = embed_text(chunks)
    print(vectors.shape)
