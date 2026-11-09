"""Read pinned Parquet columns remotely; preserve each QA trace as one unit."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq

REVISION = "10a8dd461bb6c580fa25b83cd3b2f6810f1d6385"
DATASET = "jupyter-agent/jupyter-agent-dataset"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def read_shard(fs, index, columns):
    name = f"data/non_thinking-{index:05d}-of-00103.parquet"
    path = f"datasets/{DATASET}@{REVISION}/{name}"
    with fs.open(path, "rb", block_size=262144, cache_type="readahead") as handle:
        parquet = pq.ParquetFile(handle)
        for batch in parquet.iter_batches(batch_size=32, columns=columns):
            yield from batch.to_pylist()


def run(root):
    root = Path(root)
    dest = root / "data" / "inventory-shards"
    dest.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    columns = ["id", "question", "answer", "kaggle_dataset_name", "executor_type",
               "files_used", "packages_used", "edu_score"]
    for index in range(103):
        output = dest / f"{index:03d}.jsonl"
        if output.exists():
            continue
        for attempt in range(3):
            try:
                rows = list(read_shard(fs, index, columns))
                for ordinal, row in enumerate(rows):
                    row.update(shard=index, row=ordinal,
                               notebook_id=row["id"].split("_qa_")[0],
                               metadata_sha256=digest(row))
                temp = output.with_suffix(".tmp")
                with temp.open("w") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                temp.replace(output)
                print(json.dumps(dict(shard=index, rows=len(rows),
                                      executors=dict(Counter(r["executor_type"] for r in rows)))),
                      flush=True)
                break
            except Exception as exc:
                print(json.dumps(dict(shard=index, attempt=attempt + 1,
                                      error=type(exc).__name__, detail=str(exc)[:300])), flush=True)
                if attempt == 2:
                    raise
                time.sleep(2)
    rows = [json.loads(line) for path in sorted(dest.glob("*.jsonl"))
            for line in path.open()]
    summary = dict(dataset=DATASET, revision=REVISION, split="non_thinking", rows=len(rows),
                   executors=dict(Counter(r["executor_type"] for r in rows)),
                   unique_ids=len({r["id"] for r in rows}),
                   unique_questions=len({" ".join(r["question"].casefold().split()) for r in rows}),
                   notebooks=len({r["notebook_id"] for r in rows}),
                   declared_kaggle_sources=len({r["kaggle_dataset_name"] for r in rows}),
                   source_notebooks_downloaded=False, status="inventory_complete")
    (root / "checks" / "inventory.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    run(parser.parse_args().root)
