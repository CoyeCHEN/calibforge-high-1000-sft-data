"""Fetch official benchmark assets without exposing signed download URLs."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile


def main(root):
    root = Path(root)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({
        "http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}))
    target = root / "data" / "cibench_dataset.zip"
    url = "https://github.com/open-compass/opencompass/releases/download/0.2.4.rc1/cibench_dataset.zip"
    if not target.exists():
        temp = target.with_suffix(".part")
        with opener.open(url, timeout=60) as response, temp.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        temp.replace(target)
    digest = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
    with zipfile.ZipFile(target) as archive:
        assert all(not n.startswith("/") and ".." not in Path(n).parts for n in archive.namelist())
        archive.extractall(root / "data")
        files = archive.namelist()
    result = dict(url=url, bytes=target.stat().st_size, sha256=digest,
                  members=len(files), notebooks=sum(n.endswith(".ipynb") for n in files),
                  source="Official OpenCompass release; no third-party mirror")
    (root / "checks" / "cibench_download.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    repo = "adyen/DABstep"
    meta = json.load(opener.open("https://huggingface.co/api/datasets/" + repo, timeout=30))
    (root / "checks" / "dabstep-api.json").write_text(json.dumps(meta, indent=2) + "\n")
    revision = meta["sha"]
    dest = root / "data" / "dabstep"
    dest.mkdir(exist_ok=True)
    for item in meta["siblings"]:
        name = item["rfilename"]
        if name in ("README.md", "data/tasks/dev.jsonl", "data/tasks/test.jsonl"):
            path = dest / Path(name).name
            if not path.exists():
                path.write_bytes(opener.open(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}",
                                            timeout=60).read())
            print("DABStep", revision, name, path.stat().st_size, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
