import math
import os

import numpy as np
import torch

from model import GPT, GPTConfig

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoint.pt")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "bin", "val.bin")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
BLOCK_SIZE = 256


@torch.no_grad()
def main() -> None:
    print(f"Using execution device: {DEVICE}")

    data = np.memmap(DATA_PATH, dtype=np.uint16, mode="r")
    print(f"[eval] loaded {len(data):,} validation tokens")

    model = GPT(GPTConfig())
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[eval] loaded checkpoint from step {ckpt.get('step', '?')}")
    else:
        print("[eval] no checkpoint found, using untrained model")
    model.to(DEVICE)
    model.eval()

    chunk_size = BATCH_SIZE * BLOCK_SIZE + 1
    total_loss = 0.0
    n_steps = 0

    for i in range(0, len(data) - chunk_size, chunk_size):
        chunk = data[i : i + chunk_size]
        chunk = torch.from_numpy(chunk.astype(np.int64))
        x = chunk[:-1].view(BATCH_SIZE, BLOCK_SIZE).to(DEVICE)
        y = chunk[1:].view(BATCH_SIZE, BLOCK_SIZE).to(DEVICE)

        _, loss = model(x, targets=y)
        total_loss += loss.item()
        n_steps += 1

    mean_loss = total_loss / n_steps
    perplexity = math.exp(mean_loss)

    print(f"Validation Loss: {mean_loss:.4f}")
    print(f"Validation Perplexity: {perplexity:.4f}")


if __name__ == "__main__":
    main()
