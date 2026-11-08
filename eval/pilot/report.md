# Clip selector eval

Episodes: 2 (conclusions about a stage need at least 10)

| selector | episodes | Recall@candidates | P@5 | NDCG@10 | boundary error (s) | stability | median pick length (s) | median candidate length (s) | LLM calls | LLM hours |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deep[until=candidates]@long | 2 | 0.65 | 0.00 | 0.00 | – | – | – | 49.2 | 122 | 0.33 |
| random@long | 1 | 0.17 | 0.20 | 0.12 | 14.9 | – | 77.0 | 77.0 | 0 | 0.00 |
| signal@long | 1 | 0.00 | 0.00 | 0.00 | – | – | 44.6 | 44.6 | 0 | 0.00 |
| deep@long | 1 | 0.67 | 0.20 | 0.13 | 4.5 | – | 76.9 | 47.9 | 887 | 1.08 |
| classic@long | 1 | 0.67 | 0.00 | 0.00 | – | – | 45.1 | 45.5 | 20 | 0.08 |

## Per episode

| selector | episode | Recall@candidates | P@5 | NDCG@10 | boundary error (s) | LLM calls |
| --- | --- | --- | --- | --- | --- | --- |
| deep[until=candidates]@long | source_J_W5X-tlmU8 | 0.67 | 0.00 | 0.00 | – | 55 |
| deep[until=candidates]@long | source_RBZTLptAMao | 0.62 | 0.00 | 0.00 | – | 67 |
| random@long | source_J_W5X-tlmU8 | 0.17 | 0.20 | 0.12 | 14.9 | 0 |
| signal@long | source_J_W5X-tlmU8 | 0.00 | 0.00 | 0.00 | – | 0 |
| deep@long | source_J_W5X-tlmU8 | 0.67 | 0.20 | 0.13 | 4.5 | 887 |
| classic@long | source_J_W5X-tlmU8 | 0.67 | 0.00 | 0.00 | – | 20 |
