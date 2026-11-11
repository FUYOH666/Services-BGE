# Retrieval comparison · demo_ru

Sorted by `recall@1`. Latency = median per query (embedding) and per rerank call.

| dense | sparse | mode | reranker | recall@1 | recall@5 | recall@10 | mrr@10 | ndcg@10 | code2_acc@1 | code2_acc@5 | code4_acc@1 | code4_acc@5 | embed p50 | rerank p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bge-m3 | — | dense+rerank@20 | Qwen3-Reranker-0.6B | 0.861 | 0.972 | 1.000 | 0.921 | 0.940 | 0.917 | 1.000 | 0.861 | 0.972 | 33 ms | 581 ms |
| bge-m3 | bge-m3 | hybrid_rrf+rerank@20 | Qwen3-Reranker-0.6B | 0.861 | 1.000 | 1.000 | 0.922 | 0.942 | 0.917 | 1.000 | 0.861 | 1.000 | 33 ms | 581 ms |
| Qwen3-Embedding-0.6B | — | dense+rerank@20 | Qwen3-Reranker-0.6B | 0.861 | 0.972 | 1.000 | 0.921 | 0.940 | 0.917 | 1.000 | 0.861 | 0.972 | 36 ms | 579 ms |
| Qwen3-Embedding-0.6B | bge-m3 | hybrid_rrf+rerank@20 | Qwen3-Reranker-0.6B | 0.861 | 0.972 | 1.000 | 0.921 | 0.940 | 0.917 | 1.000 | 0.861 | 0.972 | 36 ms | 579 ms |
| bge-m3 | — | dense | — | 0.806 | 0.972 | 1.000 | 0.875 | 0.906 | 0.833 | 1.000 | 0.806 | 0.972 | 32 ms | — |
| Qwen3-Embedding-0.6B | — | dense+rerank@20 | bge-reranker-v2-m3 | 0.778 | 0.972 | 1.000 | 0.871 | 0.903 | 0.806 | 0.972 | 0.778 | 0.972 | 36 ms | 148 ms |
| multilingual-e5-large | — | dense | — | 0.778 | 0.944 | 0.972 | 0.852 | 0.881 | 0.806 | 1.000 | 0.778 | 0.944 | 21 ms | — |
| bge-m3 | — | dense+rerank@20 | bge-reranker-v2-m3 | 0.750 | 0.972 | 1.000 | 0.857 | 0.893 | 0.778 | 0.972 | 0.750 | 0.972 | 32 ms | 149 ms |
| bge-m3 | bge-m3 | hybrid_rrf+rerank@20 | bge-reranker-v2-m3 | 0.750 | 0.972 | 1.000 | 0.858 | 0.894 | 0.778 | 1.000 | 0.750 | 0.972 | 32 ms | 149 ms |
| Qwen3-Embedding-0.6B | bge-m3 | hybrid_rrf+rerank@20 | bge-reranker-v2-m3 | 0.750 | 0.972 | 1.000 | 0.860 | 0.895 | 0.778 | 0.972 | 0.750 | 0.972 | 36 ms | 148 ms |
| multilingual-e5-large | — | dense+rerank@20 | bge-reranker-v2-m3 | 0.750 | 0.972 | 1.000 | 0.858 | 0.893 | 0.778 | 1.000 | 0.750 | 0.972 | 21 ms | 149 ms |
| multilingual-e5-large | bge-m3 | hybrid_rrf+rerank@20 | bge-reranker-v2-m3 | 0.750 | 0.972 | 1.000 | 0.860 | 0.895 | 0.778 | 0.972 | 0.750 | 0.972 | 21 ms | 149 ms |
| Qwen3-Embedding-0.6B | — | dense | — | 0.694 | 0.944 | 1.000 | 0.822 | 0.867 | 0.750 | 0.972 | 0.694 | 0.944 | 36 ms | — |
| Qwen3-Embedding-0.6B | bge-m3 | hybrid_rrf | — | 0.583 | 0.833 | 0.889 | 0.684 | 0.734 | 0.694 | 0.889 | 0.583 | 0.833 | 36 ms | — |
| bge-m3 | bge-m3 | hybrid_rrf | — | 0.556 | 0.806 | 0.889 | 0.660 | 0.715 | 0.639 | 0.861 | 0.556 | 0.806 | 32 ms | — |
| multilingual-e5-large | bge-m3 | hybrid_rrf | — | 0.500 | 0.778 | 0.944 | 0.645 | 0.717 | 0.667 | 0.833 | 0.500 | 0.778 | 21 ms | — |
| — | bge-m3 | sparse | — | 0.389 | 0.583 | 0.722 | 0.492 | 0.547 | 0.500 | 0.722 | 0.389 | 0.583 | 32 ms | — |
