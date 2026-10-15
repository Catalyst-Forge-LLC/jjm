# JJM / Laya Banking77 experiment

Single-task experiment. Interpret alongside label budget and data-overlap caveats.

| Route | Test accuracy | Macro F1 | Warm batch 1 p50 / p95 ms | Batch rows/s |
|---|---:|---:|---:|---:|
| tfidf | 80.8% | 0.807 | 0.4 / 0.5 | 100918.4 |
| jjm | 91.2% | 0.912 | 25.9 / 51.6 | 310.7 |
| laya | 62.5% | 0.627 | 27.0 / 40.7 | 130.8 |

Timings are local observations on the recorded device. Batch throughput is not request latency.
JJM and TF-IDF use supplied task labels. Laya receives no local model-fit labels, but uses validation labels for threshold selection.
Thresholds are selected on a separate portion of the training split. The test split is never used to choose them.
Readouts are not certified calibrated. JJM validation is left unaccepted. See summary.json and per-row JSONL for details.
