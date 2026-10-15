# JJM / Laya Banking77 experiment

Smoke experiment only. Insufficient sample to rank products.

| Route | Test accuracy | Macro F1 | Warm batch 1 p50 / p95 ms | Batch rows/s |
|---|---:|---:|---:|---:|
| tfidf | 50.0% | 0.494 | 0.4 / 0.5 | 46262.8 |
| jjm | 78.1% | 0.780 | 36.3 / 57.6 | 87.8 |
| laya | 68.8% | 0.708 | 23.9 / 34.2 | 107.3 |

Timings are local observations on the recorded device. Batch throughput is not request latency.
JJM and TF-IDF use supplied task labels. Laya receives no local model-fit labels, but uses validation labels for threshold selection.
Thresholds are selected on a separate portion of the training split. The test split is never used to choose them.
Readouts are not certified calibrated. JJM validation is left unaccepted. See summary.json and per-row JSONL for details.
