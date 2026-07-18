# GPT-Style LLM from Scratch — Build Plan

## Locked Decisions
- Attention: `torch.nn.MultiheadAttention` (not hand-rolled)
- Tokenizer: BPE via `tiktoken` (GPT-2 encoding, reuse don't retrain — retraining BPE is its own project)
- Dataset: WikiText-103 subset (~500MB-1GB slice) — swap to OpenWebText subset later if time allows
- Framework: PyTorch, CUDA-accelerated training

## Compute Strategy (Zero-Cost, No Subscriptions)
- **Local (ASUS ExpertBook P1, 32GB RAM, no CUDA GPU)**: dev + debug only. Write model code,
  data pipeline, run shape-check forward passes, sanity-train on tiny toy slice (~10k tokens)
  on CPU. Never attempt real training here — CPU training on this model size is days/weeks, not hours.
- **Kaggle Notebooks (primary training)**: free P100 GPU (16GB VRAM), ~30 GPU hours/week,
  ~12hr max session length. Better raw throughput than Colab's typical T4 for this workload.
- **Colab Free Tier (overflow/backup)**: T4 GPU (16GB VRAM), ~15-30 GPU hours/week (dynamic,
  unpublished quota), ~12hr max session. Rotate here once Kaggle's weekly quota is burned.
- **Hard requirement**: both platforms kill sessions without warning (idle timeout, disconnects,
  12hr cap). Training loop MUST checkpoint frequently and resume-from-checkpoint on restart —
  treat every session as disposable. Losing an unsaved run is the single biggest time-waster
  on free-tier compute.

## Workflow
1. Code locally on ExpertBook (model, data pipeline, config) — CPU only, no training.
2. Push to GitHub.
3. Clone repo into Kaggle Notebook, attach P100, run real training with checkpointing.
4. When Kaggle quota exhausted for the week, switch to Colab free tier with same repo.
5. Download trained checkpoint(s) back to laptop for local inference (`sample.py`) and write-up.

## Architecture Spec (small-but-real, trains in hours not days)
- d_model: 384
- n_heads: 6 (head_dim = 64)
- n_layers: 6
- context length (block_size): 256 tokens
- FFN hidden dim: 4x d_model = 1536
- vocab size: 50257 (GPT-2 tiktoken vocab)
- params: ~25-30M — big enough to show real convergence, small enough for single-GPU weekend training
- pre-norm architecture (LayerNorm before attention/FFN, not after) — cite: more stable gradients, this is what GPT-2/3 actually use, post-norm is outdated

## Repo Structure
```
gpt-scratch/
├── data/
│   ├── download.py          # fetch WikiText-103 subset
│   ├── prepare.py           # tokenize with tiktoken, save .bin shards
├── model/
│   ├── attention.py         # MHA wrapper + causal mask
│   ├── block.py              # transformer block (pre-norm, MHA, FFN, residuals)
│   ├── gpt.py                # full model: embedding, pos encoding, blocks, head
├── train.py                  # training loop, CUDA, mixed precision, checkpointing
├── config.py                  # hyperparams as dataclass, not magic numbers scattered around
├── sample.py                  # inference/generation script
├── requirements.txt
└── README.md                  # your interview story lives here
```

## Build Order (sequential, each step must run before next)
1. **Environment + data pipeline** — download WikiText-103 subset, tokenize with tiktoken, save as binary shards (memmap-able, avoids loading full dataset in RAM)
2. **Model skeleton** — embedding + positional encoding + single transformer block, forward pass shape-check only (no training yet)
3. **Full model** — stack blocks, add final LayerNorm + output head, verify param count matches spec
4. **Training loop** — AdamW, cosine LR schedule with warmup, gradient clipping, mixed precision (`torch.cuda.amp`), checkpoint every N steps (non-negotiable: must save optimizer state + step count + RNG state, and auto-resume from latest checkpoint on script start — free-tier sessions die without warning)
5. **Sanity train** — 500 steps on tiny data slice, confirm loss drops (catches bugs before burning real compute)
6. **Real training run** — full subset, track loss curve, save best checkpoint
7. **Sampling/inference** — generate text from trained checkpoint, temperature + top-k sampling
8. **Eval** — compute perplexity on held-out split, compare to published WikiText-103 baselines for your citation

## Failure Points to Watch
- **Causal masking bug**: `nn.MultiheadAttention` needs explicit `attn_mask` (upper-triangular -inf) or your model cheats by seeing future tokens — silent failure, loss looks fine, generation is garbage
- **Tokenizer/vocab mismatch**: embedding table size must exactly match tiktoken vocab size or you get index errors mid-training
- **Learning rate**: too high with this param count = loss spikes/NaN; start at 3e-4, cosine decay
- **VRAM**: batch_size × block_size × d_model scales fast — start batch_size=32, adjust based on OOM
- **Session death on free-tier GPU**: Kaggle/Colab kill sessions without warning — if checkpoint/resume isn't solid, you lose hours of quota to a single disconnect. Test resume logic on a short run BEFORE the real training run.

## Deliverable for Resume/Interview
- Loss curve plot (train + val)
- Perplexity number vs published WikiText-103 baselines
- 3-5 sample generations at different checkpoints (shows progression)
- README explaining pre-norm choice, why BPE over char-level, CUDA optimizations used