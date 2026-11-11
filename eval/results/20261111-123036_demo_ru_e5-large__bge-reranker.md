# Eval: e5-large__bge-reranker on demo_ru

2026-11-11T05:30:36+00:00 · corpus 36 · queries 36 · models {'embed': 'intfloat/multilingual-e5-large', 'sparse': 'BAAI/bge-m3', 'rerank': 'BAAI/bge-reranker-v2-m3'}

| run | recall@1 | recall@5 | recall@10 | mrr@10 | ndcg@10 | code2_acc@1 | code2_acc@5 | code4_acc@1 | code4_acc@5 |
|---|---|---|---|---|---|---|---|---|---|
| dense | 0.778 | 0.944 | 0.972 | 0.852 | 0.881 | 0.806 | 1.000 | 0.778 | 0.944 |
| sparse | 0.389 | 0.583 | 0.722 | 0.492 | 0.547 | 0.500 | 0.722 | 0.389 | 0.583 |
| hybrid_rrf | 0.500 | 0.778 | 0.944 | 0.645 | 0.717 | 0.667 | 0.833 | 0.500 | 0.778 |
| dense+rerank@20 | 0.750 | 0.972 | 1.000 | 0.858 | 0.893 | 0.778 | 1.000 | 0.750 | 0.972 |
| hybrid_rrf+rerank@20 | 0.750 | 0.972 | 1.000 | 0.860 | 0.895 | 0.778 | 0.972 | 0.750 | 0.972 |

Index 0.2 s · query embed p50 21 ms / p95 24 ms · rerank p50 149 ms / p95 169 ms
