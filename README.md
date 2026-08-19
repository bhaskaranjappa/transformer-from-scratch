# GPT-Style Transformer from Scratch

A decoder-only transformer built from first principles in PyTorch — no HF Transformers,
no shortcuts. Implements multi-head attention, layer norm, and feed-forward blocks manually
to understand next-token prediction mechanics end-to-end.

## What's implemented
- Custom BPE tokenizer (tiktoken-based)
- Multi-head self-attention from scratch
- Layer normalization, residual connections, feed-forward blocks
- Training loop with CUDA acceleration

## Training
- Dataset: WikiText-103
- Params: ~25M
- Trained on Kaggle P100

## Results
- Validation loss: 5.29 | Perplexity: ~198

This is an infrastructure-validation result, not a SOTA benchmark — the goal was proving
out the full training pipeline (tokenization → attention → loss convergence) end-to-end,
not competing with pretrained models at this scale.

## Why this exists
Most people call an API. This is what's underneath it.# transformer-from-scratch
