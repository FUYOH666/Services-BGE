# Eval: bge-m3__bge-reranker on demo_ru

2026-11-11T05:28:41+00:00 · corpus 36 · queries 36 · models {'embed': 'BAAI/bge-m3', 'sparse': 'BAAI/bge-m3', 'rerank': 'BAAI/bge-reranker-v2-m3'}

| run | recall@1 | recall@5 | recall@10 | mrr@10 | ndcg@10 | code2_acc@1 | code2_acc@5 | code4_acc@1 | code4_acc@5 |
|---|---|---|---|---|---|---|---|---|---|
| dense | 0.806 | 0.972 | 1.000 | 0.875 | 0.906 | 0.833 | 1.000 | 0.806 | 0.972 |
| sparse | 0.389 | 0.583 | 0.722 | 0.492 | 0.547 | 0.500 | 0.722 | 0.389 | 0.583 |
| hybrid_rrf | 0.556 | 0.806 | 0.889 | 0.660 | 0.715 | 0.639 | 0.861 | 0.556 | 0.806 |
| dense+rerank@20 | 0.750 | 0.972 | 1.000 | 0.857 | 0.893 | 0.778 | 0.972 | 0.750 | 0.972 |
| hybrid_rrf+rerank@20 | 0.750 | 0.972 | 1.000 | 0.858 | 0.894 | 0.778 | 1.000 | 0.750 | 0.972 |

Index 0.3 s · query embed p50 32 ms / p95 34 ms · rerank p50 149 ms / p95 169 ms
