# Clip selector eval

Episodes: 9 (conclusions about a stage need at least 10)

| selector | episodes | Recall@candidates | P@5 | NDCG@10 | boundary error (s) | stability | median pick length (s) | median candidate length (s) | LLM calls | LLM hours |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random@long | 9 | 0.04 | 0.04 | 0.05 | 19.6 | – | 69.3 | 69.3 | 0 | 0.00 |
| signal@long | 9 | 0.00 | 0.00 | 0.00 | – | – | 45.4 | 45.4 | 0 | 0.00 |
| classic@long | 9 | 0.55 | 0.16 | 0.14 | 11.8 | – | 48.1 | 45.1 | 380 | 1.66 |
| deep@long | 9 | 0.71 | 0.16 | 0.20 | 11.6 | – | 59.5 | 47.8 | 14267 | 18.55 |

## Per episode

| selector | episode | Recall@candidates | P@5 | NDCG@10 | boundary error (s) | LLM calls |
| --- | --- | --- | --- | --- | --- | --- |
| random@long | source_-AdkwqkE20M | 0.07 | 0.20 | 0.22 | 21.3 | 0 |
| signal@long | source_-AdkwqkE20M | 0.00 | 0.00 | 0.00 | – | 0 |
| classic@long | source_-AdkwqkE20M | 0.66 | 0.60 | 0.53 | 7.1 | 65 |
| deep@long | source_-AdkwqkE20M | 0.66 | 0.20 | 0.21 | 5.7 | 2378 |
| random@long | source_0AfsC-9pIo4 | 0.00 | 0.00 | 0.00 | – | 0 |
| signal@long | source_0AfsC-9pIo4 | 0.00 | 0.00 | 0.00 | – | 0 |
| classic@long | source_0AfsC-9pIo4 | 0.33 | 0.00 | 0.00 | – | 25 |
| deep@long | source_0AfsC-9pIo4 | 1.00 | 0.20 | 0.15 | 5.0 | 1064 |
| random@long | source_8son5OkbC90 | 0.11 | 0.00 | 0.07 | – | 0 |
| signal@long | source_8son5OkbC90 | 0.00 | 0.00 | 0.00 | – | 0 |
| classic@long | source_8son5OkbC90 | 0.78 | 0.20 | 0.15 | 16.1 | 41 |
| deep@long | source_8son5OkbC90 | 0.67 | 0.20 | 0.10 | 15.0 | 1424 |
| random@long | source_Lc8to_uMHVw | 0.12 | 0.00 | 0.07 | – | 0 |
| signal@long | source_Lc8to_uMHVw | 0.00 | 0.00 | 0.00 | – | 0 |
| classic@long | source_Lc8to_uMHVw | 0.62 | 0.20 | 0.17 | 3.4 | 32 |
| deep@long | source_Lc8to_uMHVw | 0.88 | 0.00 | 0.17 | – | 1373 |
| random@long | source_k7krNQaVUP8 | 0.00 | 0.00 | 0.00 | – | 0 |
| signal@long | source_k7krNQaVUP8 | 0.00 | 0.00 | 0.00 | – | 0 |
| classic@long | source_k7krNQaVUP8 | 0.38 | 0.00 | 0.00 | – | 41 |
| deep@long | source_k7krNQaVUP8 | 0.75 | 0.00 | 0.08 | – | 1454 |
| random@long | source_kCRGasHlPP8 | 0.00 | 0.00 | 0.00 | – | 0 |
| signal@long | source_kCRGasHlPP8 | 0.00 | 0.00 | 0.00 | – | 0 |
| classic@long | source_kCRGasHlPP8 | 0.75 | 0.20 | 0.16 | 13.2 | 38 |
| deep@long | source_kCRGasHlPP8 | 0.75 | 0.20 | 0.25 | 2.2 | 1457 |
| random@long | source_ky1oHHJ5Ne8 | 0.00 | 0.00 | 0.00 | – | 0 |
| signal@long | source_ky1oHHJ5Ne8 | 0.00 | 0.00 | 0.00 | – | 0 |
| classic@long | source_ky1oHHJ5Ne8 | 0.33 | 0.20 | 0.15 | 19.1 | 62 |
| deep@long | source_ky1oHHJ5Ne8 | 0.33 | 0.20 | 0.30 | 19.7 | 2285 |
| random@long | source_nulkcjbI-pI | 0.09 | 0.20 | 0.09 | 17.8 | 0 |
| signal@long | source_nulkcjbI-pI | 0.00 | 0.00 | 0.00 | – | 0 |
| classic@long | source_nulkcjbI-pI | 0.64 | 0.00 | 0.07 | – | 32 |
| deep@long | source_nulkcjbI-pI | 0.36 | 0.40 | 0.29 | 22.1 | 1337 |
| random@long | source_zB6PhTxnj7U | 0.00 | 0.00 | 0.00 | – | 0 |
| signal@long | source_zB6PhTxnj7U | 0.00 | 0.00 | 0.00 | – | 0 |
| classic@long | source_zB6PhTxnj7U | 0.50 | 0.00 | 0.00 | – | 44 |
| deep@long | source_zB6PhTxnj7U | 1.00 | 0.00 | 0.22 | – | 1495 |
