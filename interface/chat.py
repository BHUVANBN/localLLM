import torch
import sentencepiece as spm
from model.tiny_llm import TinyLLM, generate as generate_sampling
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


def generate(prompt: str, max_new_tokens: int = 50, top_k: int = 20, temperature: float = 1.0) -> str:
    return generate_sampling(model, sp, prompt, max_new_tokens=max_new_tokens, top_k=top_k, temperature=temperature, device=device)
