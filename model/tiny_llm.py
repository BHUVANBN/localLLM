import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(sz: int, device=None):
    # Creates an upper-triangular mask of -inf, with zeros on the diagonal.
    # This prevents attention to future tokens.
    mask = torch.full((sz, sz), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


class TinyLLM(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int = 128, n_heads: int = 2, n_layers: int = 2, hidden: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.pos_emb = nn.Embedding(2048, emb_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=emb_size,
                nhead=n_heads,
                dim_feedforward=hidden,
                batch_first=True,
                activation='gelu'
            ) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, vocab_size)

    def forward(self, x: torch.Tensor):
        # x: [B, T]
        B, T = x.shape
        device = x.device
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        h = self.emb(x) + self.pos_emb(pos)
        attn_mask = causal_mask(T, device=device)
        for layer in self.layers:
            h = layer(h, attn_mask=attn_mask)
        h = self.ln(h)
        logits = self.head(h)  # [B, T, V]
        return logits


def top_k_filtering(logits: torch.Tensor, top_k: int = 50):
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, top_k)
    min_values = values[..., -1, None]
    filtered = torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)
    return filtered


def generate(model: TinyLLM, sp, prompt: str, max_new_tokens: int = 50, top_k: int = 20, temperature: float = 1.0, device=None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    ids = sp.encode(prompt, out_type=int)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            next_logits = top_k_filtering(next_logits, top_k=top_k)
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [B, 1]
            input_ids = torch.cat([input_ids, next_id], dim=1)
    return sp.decode(input_ids[0].tolist())
