import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    ffn_dim: int = 1536
    bias: bool = False


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.ffn_dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.ffn_dim, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = nn.MultiheadAttention(
            config.n_embd, config.n_head, batch_first=True, bias=config.bias
        )
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

        mask = torch.triu(
            torch.full((config.block_size, config.block_size), float("-inf")),
            diagonal=1,
        )
        self.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        attn_out, _ = self.attn(
            self.ln_1(x), self.ln_1(x), self.ln_1(x),
            attn_mask=self.attn_mask[:T, :T],
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        )
        token_embd = self.wte(idx)
        pos = torch.arange(0, T, device=idx.device)
        pos_embd = self.wpe(pos)
        x = token_embd + pos_embd
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss


if __name__ == "__main__":
    config = GPTConfig()
    model = GPT(config)
    model.eval()

    B, T = 4, 256
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    with torch.no_grad():
        logits, loss = model(idx, targets=targets)

    print(f"logits shape: {logits.shape}")
    assert logits.shape == (B, T, config.vocab_size), (
        f"expected (4, 256, 50257), got {logits.shape}"
    )
    print(f"loss: {loss.item():.4f}")
    assert loss is not None and loss.ndim == 0, (
        f"expected scalar loss, got {loss}"
    )
    print("shape & loss assertion OK")
