Test set: OffensEval-2019 test split, 860 tweets (240 offensive). 
Latency: mean over 860 single-message runs on AMD Ryzen 7 5800H with Radeon Graphics, PyTorch 2.9.1+cu130, 1 CPU thread(s), batch size 1. Budget: 50 ms/message.

| Model | Precision | Macro-F1 | Accuracy | Size (MB) | Latency mean (ms) | Latency p95 (ms) | Within budget |
|---|---|---|---|---|---|---|---|
| Rule: edit-distance lexicon | rules | 0.6571 | 0.6872 | 0.002 | 7.10 | 19.93 | yes |
| Baseline: tf-idf + LogReg | FP64 | 0.7342 | 0.7767 | 3.204 | 2.40 | 3.09 | yes |
| NN: own word2vec + FFN | FP32 | 0.6639 | 0.7244 | 3.676 | 0.46 | 0.67 | yes |
| Transformer: BERT (FP32) | FP32 | 0.7985 | 0.8477 | 437.959 | 210.80 | 333.84 | no |
| Transformer: BERT (INT8 quantized) | INT8 weights | 0.7729 | 0.8419 | 181.494 | 131.22 | 193.43 | no |
| Distilled: BERT-small (FP32) | FP32 | 0.8040 | 0.8453 | 115.067 | 37.56 | 57.88 | yes |
| Distilled: BERT-small (INT8 quantized) | INT8 weights | 0.8069 | 0.8477 | 76.569 | 30.33 | 43.55 | yes |
