import faiss
import numpy as np
from embeddings.embed_chunks import embed_text
from rag.pdf_chunks import extract_chunks

def build_index(pdf_path):
    chunks = extract_chunks(pdf_path)
    vectors = embed_text(chunks).astype('float32')
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, chunks

def query_index(index, chunks, query_text, top_k=3):
    query_vector = embed_text([query_text]).astype('float32')
    D, I = index.search(query_vector, top_k)
    return [chunks[i] for i in I[0]]
