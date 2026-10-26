"""Build the Basque word frequency list shipped in psycholing_metrics/data.

wordfreq has no Basque list (it silently falls back to Spanish), so Basque
frequencies come from EusCrawl v1.1 (Artetxe et al., 2022; updated for Latxa,
Etxaniz et al., 2024), downloaded from
https://huggingface.co/datasets/HiTZ/latxa-corpus-v1.1/tree/main/euscrawl-v1.1

Tokens are counted the way wordfreq looks them up: lossy_tokenize, then digit
sequences smashed to 0. A token's frequency is its count over all tokens. As in
wordfreq's "large" lists, tokens rarer than 1e-8 are left out.

Usage:
    python scripts/build_euscrawl_frequencies.py train.jsonl.gz valid.jsonl.gz test.jsonl.gz
"""

import argparse
import gzip
import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from wordfreq import lossy_tokenize
from wordfreq.numbers import smash_numbers

LANGUAGE = "eu"
MIN_FREQUENCY = 1e-8
OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "psycholing_metrics/data/euscrawl_v1.1_eu_frequencies.tsv.gz"
)


def count_lines(lines: list) -> Counter:
    counts = Counter()
    for line in lines:
        text = json.loads(line)["text"]
        counts.update(smash_numbers(t) for t in lossy_tokenize(text, LANGUAGE))
    return counts


def batches(paths: list, size: int = 2000):
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            batch = []
            for line in f:
                batch.append(line)
                if len(batch) == size:
                    yield batch
                    batch = []
            if batch:
                yield batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="EusCrawl v1.1 jsonl.gz files")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    counts = Counter()
    with Pool(args.workers) as pool:
        for batch_counts in pool.imap_unordered(count_lines, batches(args.paths)):
            counts.update(batch_counts)

    total = sum(counts.values())
    min_count = MIN_FREQUENCY * total
    kept = sorted(
        ((word, n) for word, n in counts.items() if n >= min_count),
        key=lambda item: (-item[1], item[0]),
    )
    with gzip.open(OUTPUT, "wt", encoding="utf-8", compresslevel=9) as f:
        f.write("Word\tFrequency\n")
        for word, n in kept:
            f.write(f"{word}\t{n / total:.6g}\n")
    print(
        f"{total:,} tokens, {len(counts):,} types; kept {len(kept):,} types "
        f"with count >= {min_count:.1f} ({sum(n for _, n in kept) / total:.2%} of tokens)"
    )


if __name__ == "__main__":
    main()
