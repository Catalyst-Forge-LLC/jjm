# jjm.dev

Public alpha page for [Just Judgment Machine](https://github.com/Catalyst-Forge-LLC/jjm). [FilePress](https://getfilepress.com) (`getfilepress` on npm).

The page states the same limits as the repository README. It does not host inference, and it does not offer a package install.

```bash
pnpm install
pnpm dev
pnpm build
```

## Deploy (Cloudflare Pages)

```bash
pnpm ship
```

That builds this folder and uploads it to the Pages project `jjm`. Attach **jjm.dev** to that project. One pipeline only.

## Benchmark publication

The report is in pages/routing-benchmark.md and the CPU guide in pages/tfidf.md.
Both are linked from navigation; the report is also linked from the home page
and repository README. Published downloads live in static/benchmarks.

Rebuild the downloads with Python, without model downloads or inference:

~~~sh
python scripts/package-benchmarks.py
~~~

This verifies the immutable benchmark's 78 manifest entries and archives an
explicit allowlist from the tested CPU-backend commit. Update that pinned commit
only after rerunning integration checks. Keep original baseline results separate
from later configurations. The archive checksums are published beside the ZIPs.
