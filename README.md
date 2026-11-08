<div align="center">
  <h1>GPT-Style Transformer from Scratch</h1>
  <p><strong>Build the fundamentals behind autoregressive language models — from tokenization and causal attention to training, evaluation, and text generation.</strong></p>
  <p>
    <a href="#architecture">Architecture</a>
    ·
    <a href="#quickstart">Quickstart</a>
    ·
    <a href="#training">Training</a>
    ·
    <a href="#evaluation">Evaluation</a>
  </p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/Architecture-Decoder--Only%20GPT-7C3AED" alt="Decoder-only GPT">
    <img src="https://img.shields.io/badge/Tokenizer-GPT--2%20BPE-0F766E" alt="GPT-2 BPE">
    <img src="https://img.shields.io/badge/Dataset-WikiText--103-2563EB" alt="WikiText-103">
  </p>
</div>

---

## Overview

This repository implements a compact, decoder-only GPT-style language model in PyTorch. It is an educational engineering project focused on the mechanics of autoregressive language modeling.

The pipeline starts with raw WikiText-103 text, applies GPT-2 byte-pair encoding, and stores token IDs in binary files. A GPT-style model learns next-token prediction through causal self-attention. The repository also includes checkpointed training, validation perplexity, and autoregressive sampling.

> **Project scope:** this is a small-scale learning implementation, not a reproduction of a production GPT system. Attention uses PyTorch's `nn.MultiheadAttention`; it is not manually implemented from low-level tensor operations.

## At a glance

<div align="center">
  <table>
    <tr>
      <td align="center"><h2>6</h2><sub>Transformer layers</sub></td>
      <td align="center"><h2>6</h2><sub>Attention heads</sub></td>
      <td align="center"><h2>384</h2><sub>Embedding dimension</sub></td>
      <td align="center"><h2>256</h2><sub>Context length</sub></td>
    </tr>
  </table>
</div>

| Specification | Value |
|---|---|
| Model family | Decoder-only GPT |
| Framework | PyTorch |
| Approximate model size | ~30M parameters |
| Tokenizer | GPT-2 BPE via `tiktoken` |
| Vocabulary size | 50,257 |
| Feed-forward dimension | 1,536 |
| Dataset | WikiText-103 raw |
| Optimizer | AdamW |
| Training steps | 500 configured steps |
| Checkpoint interval | Every 100 steps |
| Execution device | CUDA when available; CPU otherwise |

## Capabilities

- **GPT-style decoder** — token and learned positional embeddings, stacked transformer blocks, final normalization, and a language-model head.
- **Causal self-attention** — an explicit upper-triangular additive mask prevents tokens from attending to future positions.
- **Pre-LayerNorm blocks** — normalization is applied before attention and feed-forward sublayers.
- **Weight tying** — input token embeddings and output projection share weights.
- **WikiText-103 pipeline** — downloads raw dataset splits and tokenizes them into compact binary files.
- **Memory-mapped batches** — reads token streams through NumPy `memmap` rather than loading entire files into RAM.
- **Checkpoint resume** — restores model weights, optimizer state, training step, and data-loader position.
- **Evaluation and sampling** — calculates validation loss/perplexity and generates text using temperature and top-k sampling.

<a id="architecture"></a>
## Architecture

The model is a decoder-only Transformer. Input token embeddings are combined with learned positional embeddings, then passed through six pre-normalized transformer blocks.

### 1. End-to-end model

~~~mermaid
flowchart TB
    TOK["Input token IDs"] --> EMB["Token embedding"]
    POS["Position IDs"] --> PEMB["Learned positional embedding"]
    EMB --> ADD["Embedding sum"]
    PEMB --> ADD
    ADD --> BLOCKS["Transformer blocks × 6"]
    BLOCKS --> LN["Final LayerNorm"]
    LN --> HEAD["Weight-tied language-model head"]
    HEAD --> LOGITS["Next-token logits"]
    LOGITS --> LOSS["Cross-entropy loss<br/>during training"]
    LOGITS --> SAMPLE["Autoregressive sampling<br/>during inference"]

    classDef input fill:#EEF2FF,stroke:#6366F1,color:#1E1B4B,stroke-width:1.5px
    classDef process fill:#F8FAFC,stroke:#94A3B8,color:#0F172A,stroke-width:1.2px
    classDef model fill:#F5F3FF,stroke:#8B5CF6,color:#2E1065,stroke-width:1.5px
    classDef output fill:#ECFDF5,stroke:#10B981,color:#064E3B,stroke-width:1.5px

    class TOK,POS input
    class EMB,PEMB,ADD,LN process
    class BLOCKS,HEAD model
    class LOGITS,LOSS,SAMPLE output
~~~

### 2. Transformer block

Each block contains two residual sublayers: causal multi-head self-attention and a position-wise feed-forward network.

~~~mermaid
flowchart TB
    X["Input hidden states"] --> LN1["LayerNorm"]
    LN1 --> ATT["Causal multi-head attention"]
    ATT --> ADD1["Residual addition"]
    X --> ADD1
    ADD1 --> LN2["LayerNorm"]
    LN2 --> MLP["Linear → GELU → Linear"]
    MLP --> ADD2["Residual addition"]
    ADD1 --> ADD2
    ADD2 --> OUT["Output hidden states"]

    classDef input fill:#EEF2FF,stroke:#6366F1,color:#1E1B4B,stroke-width:1.5px
    classDef process fill:#F8FAFC,stroke:#94A3B8,color:#0F172A,stroke-width:1.2px
    classDef attention fill:#ECFEFF,stroke:#0891B2,color:#164E63,stroke-width:1.5px
    classDef output fill:#ECFDF5,stroke:#10B981,color:#064E3B,stroke-width:1.5px

    class X input
    class LN1,LN2,ADD1,ADD2 process
    class ATT,MLP attention
    class OUT output
~~~

**Attention implementation:** `model.py` delegates the attention operation to `torch.nn.MultiheadAttention` and supplies an additive causal mask. The query, key, value projections and attention kernel are therefore managed by PyTorch.

### 3. Training and inference lifecycle

~~~mermaid
flowchart LR
    RAW["WikiText-103"] --> DOWNLOAD["Download raw splits"]
    DOWNLOAD --> TOKENIZE["GPT-2 BPE tokenization"]
    TOKENIZE --> BIN["uint16 binary token files"]
    BIN --> TRAIN["Next-token training"]
    TRAIN --> CKPT["checkpoint.pt"]
    CKPT --> EVAL["Validation loss & perplexity"]
    CKPT --> GEN["Text generation"]

    classDef data fill:#EEF2FF,stroke:#6366F1,color:#1E1B4B,stroke-width:1.5px
    classDef process fill:#F8FAFC,stroke:#94A3B8,color:#0F172A,stroke-width:1.2px
    classDef artifact fill:#ECFEFF,stroke:#0891B2,color:#164E63,stroke-width:1.5px
    classDef result fill:#ECFDF5,stroke:#10B981,color:#064E3B,stroke-width:1.5px

    class RAW,BIN data
    class DOWNLOAD,TOKENIZE,TRAIN process
    class CKPT artifact
    class EVAL,GEN result
~~~

## Technology stack

| Layer | Technology |
|---|---|
| Model implementation | PyTorch |
| Tokenization | tiktoken — GPT-2 encoding |
| Dataset loading | Hugging Face Datasets |
| Dataset processing | NumPy, Python binary I/O |
| Optimizer | PyTorch AdamW |
| Data access | NumPy memory mapping |
| Evaluation | Cross-entropy loss, perplexity |
| Text generation | Temperature and top-k sampling |

## Repository structure

~~~text
transformer-from-scratch/
├── data/
│   ├── download.py       # Download WikiText-103 raw splits
│   └── prepare.py        # Tokenize text and write binary token files
├── model.py              # GPTConfig, MLP, Block, and GPT
├── train.py              # Data loader, training loop, checkpointing
├── eval.py               # Validation loss and perplexity
├── sample.py             # Autoregressive text generation
├── requirements.txt      # Python dependencies
├── plan.md               # Design decisions and implementation plan
└── README.md
~~~

Runtime-generated files are not part of the source tree:

~~~text
data/
├── wikitext-103-raw/
│   ├── train.raw
│   ├── validation.raw
│   └── test.raw
└── bin/
    ├── train.bin
    ├── val.bin
    ├── test.bin
    └── meta.txt

checkpoint.pt
~~~

## Quickstart

<a id="quickstart"></a>
### Prerequisites

- Python 3.10 or newer.
- Git.
- A CUDA-capable NVIDIA GPU for practical training speed; CPU execution is supported for development and small checks.
- Internet access for the initial dataset download.

### 1. Clone the repository

~~~bash
git clone https://github.com/bhaskaranjappa/transformer-from-scratch.git
cd transformer-from-scratch
~~~

### 2. Create a virtual environment

**Windows PowerShell**

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
~~~

**Linux / macOS**

~~~bash
python -m venv .venv
source .venv/bin/activate
~~~

### 3. Install dependencies

~~~bash
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

For GPU execution, install the PyTorch build matching your CUDA setup. Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

### 4. Download the dataset

~~~bash
python data/download.py
~~~

The script downloads the `Salesforce/wikitext` dataset configuration `wikitext-103-raw-v1` and writes the train, validation, and test raw text splits.

### 5. Prepare tokenized binary data

~~~bash
python data/prepare.py
~~~

The preparation script uses GPT-2 BPE through `tiktoken`. It writes token IDs as unsigned 16-bit integers and records vocabulary metadata in `meta.txt`.

### 6. Train the model

~~~bash
python train.py
~~~

The script selects CUDA when available and CPU otherwise. It creates `checkpoint.pt` and saves checkpoints every 100 steps, including the final configured step.

### 7. Evaluate the checkpoint

~~~bash
python eval.py
~~~

### 8. Generate text

~~~bash
python sample.py
~~~

The sampling script loads `checkpoint.pt` when available. It uses the prompt `The Transformer architecture`, temperature 0.8, top-k 40, and generates up to 64 new tokens.

> **Note:** if no checkpoint exists, `sample.py` initializes a randomly weighted model. Its output is not meaningful until training has produced a checkpoint.

## Training

<a id="training"></a>
### Model configuration

The model architecture is defined by the `GPTConfig` dataclass in `model.py`.

| Parameter | Value |
|---|---:|
| Vocabulary size | 50,257 |
| Context length | 256 |
| Layers | 6 |
| Attention heads | 6 |
| Embedding dimension | 384 |
| Head dimension | 64 |
| Feed-forward dimension | 1,536 |
| Attention bias | Disabled |
| Positional embeddings | Learned |
| Output weights | Tied to token embeddings |

### Optimization settings

The current training settings are constants in `train.py`.

| Hyperparameter | Value |
|---|---:|
| Batch size | 12 |
| Block size | 256 |
| Learning rate | 6e-4 |
| Weight decay | 0.1 |
| AdamW betas | (0.9, 0.95) |
| Total steps | 500 |
| Checkpoint interval | 100 steps |

The training objective is next-token prediction. For each batch, the target sequence is the input sequence shifted by one token. Cross-entropy loss is calculated over the vocabulary at each position.

### Checkpoint behavior

The checkpoint stores:

- Model state dictionary.
- Optimizer state dictionary.
- Current training step.
- Most recently recorded loss.
- Data-loader position.

When `checkpoint.pt` exists, `train.py` restores these values and continues from the saved step. Random-number-generator states are not currently stored, so resumed runs are not guaranteed to be bit-for-bit reproducible.

## Evaluation

<a id="evaluation"></a>
The evaluation script reads the prepared validation token stream, computes mean batch cross-entropy loss, and converts it to perplexity:

```text
perplexity = exp(mean_validation_loss)
```

### Previously reported result

The repository's earlier README recorded this result:

<div align="center">
  <table>
    <tr>
      <td align="center"><h2>5.29</h2><sub>Validation loss</sub></td>
      <td align="center"><h2>~198</h2><sub>Validation perplexity</sub></td>
    </tr>
  </table>
</div>

<sub>These values are previously reported project results, not independently reproduced by this documentation update. The repository does not currently include a committed checkpoint, training log, or loss curve for independent verification.</sub>

This result should be treated as a project-run reference, not a benchmark claim against pretrained language models. Reproduction requires the same dataset version, preprocessing, checkpoint, and evaluation procedure.

## Engineering decisions

| Decision | Rationale |
|---|---|
| Decoder-only GPT | Directly models autoregressive next-token prediction. |
| GPT-2 BPE | Reuses a fixed vocabulary of 50,257 tokens. |
| Causal attention mask | Blocks information flow from future positions. |
| Pre-LayerNorm | Applies normalization before attention and MLP sublayers. |
| Weight tying | Shares input embedding and output projection parameters. |
| `uint16` token files | Stores token IDs compactly because the vocabulary fits below 65,536. |
| Memory-mapped data | Avoids loading the complete token stream into RAM. |
| Checkpoint resume | Makes training recoverable after interrupted sessions. |

## Current limitations

This implementation prioritizes clarity and learning over large-scale training efficiency.

- **Attention:** uses PyTorch's `nn.MultiheadAttention`; attention is not implemented manually from matrix operations.
- **Optimization:** no learning-rate scheduler, warmup, gradient clipping, dropout, or mixed-precision training is configured.
- **Reproducibility:** checkpoints do not preserve random-number-generator states.
- **Batching:** the loader reads sequential chunks from one flat token stream and wraps at the end. It does not perform shuffled, document-aware sampling.
- **Evaluation:** reports average validation loss and perplexity without benchmark comparisons or confidence intervals.
- **Testing:** no automated test suite or committed reproducible training log is currently included.

These are known scope boundaries, not claims of production readiness.

## Roadmap

Potential engineering extensions:

1. Add tests for causal masking, tensor shapes, loss computation, and checkpoint restoration.
2. Add train/validation loss logging and reproducible learning curves.
3. Implement warmup, cosine learning-rate decay, gradient clipping, and mixed precision.
4. Preserve RNG states for more reproducible checkpoint resumption.
5. Implement manual scaled dot-product attention as a separate educational path and compare it with PyTorch's module.
6. Add document-aware batching and more rigorous evaluation protocols.

## Learning outcomes

The project provides hands-on practice with:

- Decoder-only Transformer architecture.
- Token and positional embeddings.
- Causal multi-head self-attention.
- Residual connections and pre-LayerNorm.
- Feed-forward networks and GELU.
- Autoregressive next-token prediction.
- BPE tokenization and binary data preparation.
- AdamW optimization and checkpointing.
- Temperature and top-k sampling.
- Validation loss and perplexity.

## License

No explicit license file is currently present. Public visibility does not automatically grant permission to reuse, modify, or redistribute this repository. Add a license before inviting external reuse.

---

<div align="center">
  <p><strong>Built by <a href="https://github.com/bhaskaranjappa">Bhaskar Anjappa</a></strong></p>
  <p>
    <a href="https://github.com/bhaskaranjappa/transformer-from-scratch">Explore the source</a>
    ·
    <a href="https://github.com/bhaskaranjappa">GitHub profile</a>
  </p>
  <sub>Understanding the architecture behind next-token prediction, one component at a time.</sub>
</div>
