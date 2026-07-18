import os
import sys

import numpy as np
import torch

from model import GPT, GPTConfig

if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.isdir("/kaggle/working"):
    CHECKPOINT_PATH = "/kaggle/working/checkpoint.pt"
else:
    CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoint.pt")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "bin", "train.bin")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 12
BLOCK_SIZE = 256
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)


class ShardDataLoader:
    def __init__(self, path: str, batch_size: int, block_size: int):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.batch_size = batch_size
        self.block_size = block_size
        self.current_position = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        chunk_size = self.batch_size * self.block_size + 1
        if self.current_position + chunk_size > len(self.data):
            self.current_position = 0
        chunk = self.data[self.current_position : self.current_position + chunk_size]
        chunk = torch.from_numpy(chunk.astype(np.int64))
        x = chunk[:-1].view(self.batch_size, self.block_size)
        y = chunk[1:].view(self.batch_size, self.block_size)
        self.current_position += chunk_size
        return x, y


def save_checkpoint(model, optimizer, step, loss, loader, path: str) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "loss": loss,
            "current_position": loader.current_position,
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer, loader):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    step = ckpt["step"]
    loader.current_position = ckpt["current_position"]
    return step


def main() -> None:
    print(f"Using execution device: {DEVICE}")
    loader = ShardDataLoader(DATA_PATH, BATCH_SIZE, BLOCK_SIZE)
    model = GPT(GPTConfig())
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=BETAS
    )

    start_step = 0
    if os.path.exists(CHECKPOINT_PATH):
        start_step = load_checkpoint(CHECKPOINT_PATH, model, optimizer, loader)
        print(f"[train] resumed from checkpoint at step {start_step}")
    else:
        print("[train] no checkpoint found, starting fresh")

    model.train()
    total_steps = 500
    save_interval = 100

    for step in range(start_step, total_steps):
        x, y = loader.next_batch()
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits, loss = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"step {step} loss {loss.item():.4f}")

        if (step + 1) % save_interval == 0 or step == total_steps - 1:
            save_checkpoint(model, optimizer, step, loss.item(), loader, CHECKPOINT_PATH)
            print(f"[train] checkpoint saved at step {step}")


if __name__ == "__main__":
    main()
