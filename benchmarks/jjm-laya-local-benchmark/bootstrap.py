"""Fetch pinned source and public data. Run before installing requirements."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent
PINS = json.loads((ROOT / "pins.json").read_text())
DATA_COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"
HASHES = {"train": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
          "test": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"}

def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()

def main():
    target = ROOT / "vendor" / "jjm"
    if not target.exists():
        raw = fetch(f"https://codeload.github.com/Catalyst-Forge-LLC/jjm/tar.gz/{PINS['jjm_commit']}")
        parent = target.parent
        parent.mkdir(exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            archive.extractall(parent, filter="data")
        (parent / f"jjm-{PINS['jjm_commit']}").rename(target)
        (target / "BENCHMARK_PIN").write_text(PINS["jjm_commit"])
    elif not (target / "BENCHMARK_PIN").exists() or (target / "BENCHMARK_PIN").read_text() != PINS["jjm_commit"]:
        raise RuntimeError("Existing vendor/jjm has no matching pin. Move it aside before bootstrapping.")
    data = ROOT / "data" / "banking77"
    data.mkdir(parents=True, exist_ok=True)
    for split, expected in HASHES.items():
        path = data / f"{split}.csv"
        raw = path.read_bytes() if path.exists() else fetch(
            f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{DATA_COMMIT}/banking_data/{split}.csv")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RuntimeError(f"Dataset digest mismatch: {path}")
        path.write_bytes(raw)
    print("Pinned JJM and verified Banking77 data are ready.")

if __name__ == "__main__":
    main()
