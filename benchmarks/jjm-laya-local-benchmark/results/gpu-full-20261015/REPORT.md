# JJM / Laya Banking77 experiment

Single-task experiment. Interpret alongside label budget and data-overlap caveats.

| Route | Test accuracy | Macro F1 | Warm batch 1 p50 / p95 ms | Batch rows/s |
|---|---:|---:|---:|---:|
| tfidf | 95.6% | 0.951 | 0.4 / 0.5 | 99644.1 |
| jjm | 96.7% | 0.962 | 36.4 / 68.7 | 326.5 |
| laya | 55.3% | 0.566 | 38.9 / 78.6 | 118.6 |

Timings are local observations on the recorded device. Batch throughput is not request latency.
JJM and TF-IDF use supplied task labels. Laya receives no local model-fit labels, but uses validation labels for threshold selection.
Thresholds are selected on a separate portion of the training split. The test split is never used to choose them.
Readouts are not certified calibrated. JJM validation is left unaccepted. See summary.json and per-row JSONL for details.
