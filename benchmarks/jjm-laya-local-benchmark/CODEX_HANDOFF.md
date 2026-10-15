# Continue this experiment on the local GPU

Goal: compare my JJM project against Laya, then establish whether wrapping Laya in JJM helps my Exec Foundry workflows.

Read README.md, pins.json and the included CPU results before acting. Set up the isolated environment in this directory. Verify CUDA is available in that environment, run the integrity tests, then execute the documented 512-label/400-test and full-test GPU experiments. Record the actual hardware, package versions and any setup changes.

Keep supplied-label and actual-fit budgets explicit. Do not present Laya's pretrained run as equal-supervision to a task-fitted JJM head. Preserve the fixed train/validation/test split and reject truncation. Do not tune on test data. Do not count cached embeddings as fresh inference. Do not call normalized readouts calibrated probabilities.

Check adapter parity and inspect disagreement examples. Explain accuracy, latency, batch throughput, setup/training costs and selective coverage together. If a model fails, report the actual failure and fix the harness before comparing results. Use fresh output directories so failed and completed evidence remains inspectable.

Once the baseline is valid, try batch-size changes and one optimization at a time, rerunning parity and quality checks. The experimental adapter lives in laya_adapter.py and is programmatically registered in bench.py. It supports choice only. Do not claim it is already an upstream JJM integration.

For an EF pilot, inspect my local EF/JJM repositories and their AGENTS.md instructions. Propose a small adapter integration and task-specific evaluation grounded in the actual code. Public Banking77 results do not demonstrate job-ranking quality. Keep private job/candidate data local. Do not publish a release, push commits or modify live EF state as part of this benchmark.
