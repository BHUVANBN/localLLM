import numpy as np
import torch
import sentencepiece as spm

from model.tiny_llm import TinyLLM

_sp = None
_model = None


def _load_components():
    global _sp, _model
    if _sp is None:
        _sp = spm.SentencePieceProcessor()
        _sp.load("tokenizer/mytok.model")
    if _model is None:
        vocab_size = _sp.get_piece_size()
        device = torch.device("cpu")
        _model = TinyLLM(vocab_size).to(device)
        state = torch.load("model/tiny_llm.pt", map_location=device)
        _model.load_state_dict(state)
        _model.eval()


def embed_text(text_list):
    _load_components()
    device = torch.device("cpu")
    embs = []
    with torch.no_grad():
        for text in text_list:
            ids = _sp.encode(text, out_type=int)
            if len(ids) == 0:
                ids = [0]
            token_embs = _model.emb(torch.tensor(ids, dtype=torch.long, device=device))
            mean_emb = token_embs.mean(dim=0).cpu().numpy().astype("float32")
            embs.append(mean_emb)
    return np.stack(embs, axis=0)


if __name__ == "__main__":
    from rag.pdf_chunks import extract_chunks
    chunks = extract_chunks("data/pdfs/sample.pdf")
    vectors = embed_text(chunks)
    print(vectors.shape)
