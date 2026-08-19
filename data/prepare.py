import argparse
import os
import struct

import tiktoken

# GPT-2 BPE encoding reused (do not retrain BPE — locked decision).
ENCODING_NAME = "gpt2"
VOCAB_SIZE = 50257  # must match embedding table size in the model

# Each shard is written as uint16 token ids concatenated into a flat .bin file.
# 50257 < 65535 so uint16 is sufficient and halves disk/RAM vs uint32.
DTYPE = "H"  # unsigned short


def tokenize_file(path: str, enc) -> list:
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.extend(enc.encode(line))
    return ids


def write_shard(ids: list, path: str) -> None:
    with open(path, "wb") as f:
        for tid in ids:
            f.write(struct.pack(DTYPE, tid))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize WikiText-103 raw with tiktoken and save .bin shards"
    )
    parser.add_argument(
        "--raw-dir",
        default=os.path.join(os.path.dirname(__file__), "wikitext-103-raw"),
        help="directory containing wiki.train/valid/test.raw",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "bin"),
        help="directory to write token .bin shards",
    )
    args = parser.parse_args()

    enc = tiktoken.get_encoding(ENCODING_NAME)
    assert enc.n_vocab == VOCAB_SIZE, (
        f"vocab mismatch: tiktoken reports {enc.n_vocab}, expected {VOCAB_SIZE}"
    )

    os.makedirs(args.out_dir, exist_ok=True)

    splits = {
        "train": "train.raw",
        "val": "validation.raw",
        "test": "test.raw",
    }

    for split, fname in splits.items():
        src = os.path.join(args.raw_dir, fname)
        if not os.path.exists(src):
            print(f"[prepare] skipping missing {src}")
            continue
        print(f"[prepare] tokenizing {split} ({src})")
        ids = tokenize_file(src, enc)
        out_path = os.path.join(args.out_dir, f"{split}.bin")
        write_shard(ids, out_path)
        print(
            f"[prepare] wrote {len(ids):,} tokens -> {out_path} "
            f"({os.path.getsize(out_path) / 1e6:.1f} MB)"
        )

    # Save vocab size so the data loader / model can assert against it.
    with open(os.path.join(args.out_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"vocab_size={VOCAB_SIZE}\n")
        f.write(f"encoding={ENCODING_NAME}\n")
        f.write(f"dtype={DTYPE}\n")

    print("[prepare] done")


if __name__ == "__main__":
    main()
