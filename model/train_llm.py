import torch
import torch.nn as nn

class TinyLLM(nn.Module):
    def __init__(self, vocab_size, emb_size=128, n_heads=2, n_layers=2, hidden=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=emb_size, nhead=n_heads, dim_feedforward=hidden)
            for _ in range(n_layers)
        ])
        self.fc = nn.Linear(emb_size, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.fc(x)

if __name__ == "__main__":
    vocab_size = 8000  # same as tokenizer
    model = TinyLLM(vocab_size)
    torch.save(model.state_dict(), "model/tiny_llm.pt")
    print("Tiny LLM saved: model/tiny_llm.pt")
