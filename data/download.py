import argparse
import os

from datasets import load_dataset

# WikiText-103 raw via HuggingFace `datasets` (maintained source; the legacy
# S3 mirror is stale / 301-redirects). We pull the raw release and hand the
# text to tiktoken in prepare.py.
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
SPLITS = ("train", "validation", "test")


def download(out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    print(f"[download] loading {DATASET_NAME}/{DATASET_CONFIG} from HuggingFace")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG)

    written = {}
    for split in SPLITS:
        if split not in ds:
            print(f"[download] split '{split}' not present, skipping")
            continue
        out_path = os.path.join(out_dir, f"{split}.raw")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in ds[split]:
                text = row.get("text", "")
                # Keep the original line (incl. blank lines as empty) so the
                # token stream preserves document/article boundaries.
                f.write(text if text.endswith("\n") else text + "\n")
        written[split] = out_path
        print(
            f"[download] wrote {len(ds[split]):,} rows -> {out_path} "
            f"({os.path.getsize(out_path) / 1e6:.1f} MB)"
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Download WikiText-103 raw via HuggingFace datasets")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "wikitext-103-raw"),
        help="directory to store the raw .raw files",
    )
    args = parser.parse_args()

    written = download(args.out_dir)
    if not written:
        print("[download] WARNING: no splits were written")
    else:
        print("[download] done")


if __name__ == "__main__":
    main()
