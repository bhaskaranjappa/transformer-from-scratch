import os

import tiktoken
import torch

from model import GPT, GPTConfig

if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.isdir("/kaggle/working"):
    CHECKPOINT_PATH = "/kaggle/working/checkpoint.pt"
else:
    CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoint.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def main() -> None:
    print(f"Using execution device: {DEVICE}")

    if not os.path.exists(CHECKPOINT_PATH):
        print("[sample] no checkpoint found, using untrained model")
        model = GPT(GPTConfig())
    else:
        print("[sample] loading checkpoint")
        model = GPT(GPTConfig())
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[sample] loaded checkpoint from step {ckpt.get('step', '?')}")

    model.to(DEVICE)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    prompt = "The Transformer architecture"
    idx = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).to(DEVICE)

    output = generate(model, idx, max_new_tokens=64, temperature=0.8, top_k=40)
    print(enc.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
