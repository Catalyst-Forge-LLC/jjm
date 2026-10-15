"""Rebuild the published evidence downloads using only verified, public files.

Run from any directory with Python 3.10+. The original measured files must
still match their manifest. The source ZIP is an explicit allowlist from a
fixed local Git commit; environments, model caches, and private data are excluded.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmarks/jjm-laya-local-benchmark"
DEST = ROOT / "site/static/benchmarks"
SOURCE = "6adf62d6912a1a78ed0d1bdd89ad4a4889b0b8e8"


def main():
    manifest = json.loads((BENCH/"SHA256SUMS.json").read_text(encoding="utf-8"))
    contents = {}
    for name, expected in sorted(manifest.items()):
        raw = (BENCH/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Evidence hash mismatch: {name}")
        contents["jjm-laya-local-benchmark/"+name] = raw
    contents["jjm-laya-local-benchmark/SHA256SUMS.json"] = (BENCH/"SHA256SUMS.json").read_bytes()
    DEST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DEST/"routing-2026-10-15.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, raw in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026,10,15,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, raw)
    subprocess.run(["git", "archive", "--format=zip", f"--output={DEST / 'jjm-tfidf-6adf62d.zip'}",
                    SOURCE, "src", "examples", "tests", "specs", "pyproject.toml", "uv.lock",
                    "README.md", "LICENSE", "NOTICE", "trials/routing",
                    "benchmarks/jjm-laya-local-benchmark", ".github/workflows/test.yml"],
                   cwd=ROOT, check=True)
    (DEST/"tfidf-backend-summary.json").write_bytes(
        (ROOT/"trials/routing/results/tfidf-backend-20261015/summary.json").read_bytes())
    names = ["routing-2026-10-15.zip", "jjm-tfidf-6adf62d.zip", "tfidf-backend-summary.json"]
    (DEST/"SHA256SUMS.txt").write_text("".join(
        f"{hashlib.sha256((DEST/name).read_bytes()).hexdigest()}  {name}\n" for name in names),
        encoding="utf-8", newline="\n")
    print(f"Verified {len(manifest)} evidence files and packaged {len(names)} public downloads.")


if __name__ == "__main__":
    main()
