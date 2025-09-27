import torch
import torch.nn as nn
import torch.optim as optim
import sentencepiece as spm
from pathlib import Path
from typing import List

from model.tiny_llm import TinyLLM


def load_tokens(sp: spm.SentencePieceProcessor, filepath: str) -> List[int]:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    ids = sp.encode(text, out_type=int)
    return ids


def make_batches(ids: List[int], block_size: int, batch_size: int):
    import math
    # Create many sequences of length block_size for inputs and targets shifted by 1
    n_tokens = len(ids)
    n_blocks = (n_tokens - 1) // block_size
    X, Y = [], []
    for i in range(n_blocks):
        start = i * block_size
        x = ids[start:start + block_size]
        y = ids[start + 1:start + 1 + block_size]
        if len(x) == block_size and len(y) == block_size:
            X.append(x)
            Y.append(y)
    # yield mini-batches
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size], dtype=torch.long)
        yb = torch.tensor(Y[i:i + batch_size], dtype=torch.long)
        yield xb, yb


def train(
    data_path: str = "data/train.txt",
    tokenizer_path: str = "tokenizer/mytok.model",
    save_path: str = "model/tiny_llm.pt",
    emb_size: int = 128,
    n_heads: int = 2,
    n_layers: int = 2,
    hidden: int = 256,
    block_size: int = 64,
    batch_size: int = 16,
    epochs: int = 3,
    lr: float = 3e-4,
):
    device = torch.device("cpu")
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    vocab_size = sp.get_piece_size()

    model = TinyLLM(vocab_size, emb_size=emb_size, n_heads=n_heads, n_layers=n_layers, hidden=hidden).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    ids = load_tokens(sp, data_path)
    if len(ids) < block_size + 1:
        raise ValueError("Training text is too short for the chosen block_size.")

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_steps = 0
        for xb, yb in make_batches(ids, block_size, batch_size):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)  # [B, T, V]
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_steps += 1
        avg = total_loss / max(n_steps, 1)
        print(f"Epoch {epoch+1}/{epochs} - loss: {avg:.4f}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train TinyLLM with next-token prediction")
    parser.add_argument("--data_path", type=str, default="data/train.txt")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer/mytok.model")
    parser.add_argument("--save_path", type=str, default="model/tiny_llm.pt")
    parser.add_argument("--emb_size", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    train(
        data_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        save_path=args.save_path,
        emb_size=args.emb_size,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        hidden=args.hidden,
        block_size=args.block_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )
