# Задача: обновить скиллы и правила Cursor на Linux-сервере

## Контекст

На MacBook подняты **нативные** сервисы BGE-M3 (embedding) и BGE-Reranker-v2-m3 (reranker) вместо LM Studio. Серверный Cursor должен обновить свои skills и rules, чтобы обращаться к ним по правильным адресам.

## Что изменилось

| Параметр | Было (LM Studio) | Стало (Services-BGE) |
|----------|-------------------|----------------------|
| Embedding URL | `http://100.75.152.22:1234` | `http://100.75.152.22:9001` |
| Embedding model | `text-embedding-bge-m3` | `BAAI/bge-m3` |
| Embedding API | только dense (`/v1/embeddings`) | dense + sparse + colbert (`/v1/embeddings`) |
| Reranker URL | `http://100.75.152.22:1234` | `http://100.75.152.22:9002` |
| Reranker model | `text-embedding-bge-reranker-v2-m3` | `BAAI/bge-reranker-v2-m3` |
| Reranker API | embedding + cosine hack | нативный `/v1/rerank` (cross-encoder) |
| Health endpoint | `GET /v1/models` | `GET /healthz` |

## Что нужно сделать

### 1. Просканировать доступность сервисов

Проверь, что MacBook-сервисы доступны через TailScale:

```bash
curl -s http://100.75.152.22:9001/healthz
# Ожидаем: {"status":"ok","model":"BAAI/bge-m3","device":"mps","ready":true}

curl -s http://100.75.152.22:9002/healthz
# Ожидаем: {"status":"ok","model":"BAAI/bge-reranker-v2-m3","device":"mps","ready":true}
```

### 2. Обновить переменные окружения

Во всех `.env` файлах проектов на сервере заменить:

```bash
# Было:
LOCAL_AI_EMBEDDING_BASE_URL=http://100.75.152.22:1234
LOCAL_AI_EMBEDDING_MODEL=text-embedding-bge-m3
LOCAL_AI_RERANKER_BASE_URL=http://100.75.152.22:1234
LOCAL_AI_RERANKER_MODEL=text-embedding-bge-reranker-v2-m3

# Стало:
LOCAL_AI_EMBEDDING_BASE_URL=http://100.75.152.22:9001
LOCAL_AI_EMBEDDING_MODEL=BAAI/bge-m3
LOCAL_AI_RERANKER_BASE_URL=http://100.75.152.22:9002
LOCAL_AI_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

### 3. Обновить Cursor skills

Найди и обнови все skills, которые ссылаются на embedding или reranker:

- `~/.cursor/skills/remote-embedding-service/SKILL.md`
- `~/.cursor/skills/remote-reranker-service/SKILL.md`
- `~/.cursor/skills/local-ai-services-base/SKILL.md`

**Ключевые замены в embedding skill:**
- URL: `127.0.0.1:1234` → `100.75.152.22:9001`
- Health: `GET /v1/models` → `GET /healthz`
- Убрать упоминания LM Studio
- Добавить: ответ теперь содержит `sparse_embedding` и `sparse_qdrant` (для Qdrant hybrid search)

**Ключевые замены в reranker skill:**
- URL: `127.0.0.1:1234` → `100.75.152.22:9002`
- Health: `GET /v1/models` → `GET /healthz`
- Убрать cosine similarity workaround — теперь нативный `POST /v1/rerank`
- Убрать упоминания LM Studio

### 4. Обновить Cursor rules

Найди и обнови `macbook-overview.mdc` или аналогичный файл:

- `~/.cursor/rules/macbook-overview.mdc`
- Любой rule с упоминанием порта 1234, LM Studio embedding/reranker

**Замены:**
- Порт `1234` (LM Studio) → порты `9001` (embedding) и `9002` (reranker)
- `text-embedding-bge-m3` → `BAAI/bge-m3`
- `text-embedding-bge-reranker-v2-m3` → `BAAI/bge-reranker-v2-m3`

### 5. Обновить local-ai-services-base

В `~/.cursor/skills/local-ai-services-base/SKILL.md`:
- Убрать секцию `lm_studio` из конфигурации
- Добавить секции `embedding` (порт 9001) и `reranker` (порт 9002)
- Обновить список переменных окружения

## API reference (для обновления skills)

### Embedding — POST /v1/embeddings

```json
// Request
{
  "input": ["текст 1", "текст 2"],
  "return_dense": true,
  "return_sparse": true,
  "return_colbert": false,
  "max_length": 512
}

// Response
{
  "object": "list",
  "data": [
    {
      "index": 0,
      "dense_embedding": [0.123, -0.456, ...],
      "sparse_embedding": {"трубы": 0.89, "стали": 0.52, ...},
      "sparse_qdrant": {"indices": [1234, 5678], "values": [0.89, 0.52]},
      "colbert_embedding": null
    }
  ],
  "model": "BAAI/bge-m3",
  "usage": {"total_tokens": 0}
}
```

### Reranker — POST /v1/rerank

```json
// Request
{
  "query": "search query",
  "documents": ["doc 1", "doc 2"],
  "top_n": 5,
  "normalize": true
}

// Response
{
  "results": [
    {"index": 0, "document": "doc 1", "relevance_score": 0.95},
    {"index": 1, "document": "doc 2", "relevance_score": 0.12}
  ],
  "model": "BAAI/bge-reranker-v2-m3",
  "usage": {"total_pairs": 2}
}
```

## Чек-лист

- [ ] `curl http://100.75.152.22:9001/healthz` — embedding доступен
- [ ] `curl http://100.75.152.22:9002/healthz` — reranker доступен
- [ ] `.env` файлы обновлены (все проекты на сервере)
- [ ] `remote-embedding-service` skill обновлён
- [ ] `remote-reranker-service` skill обновлён
- [ ] `local-ai-services-base` skill обновлён
- [ ] `macbook-overview` rule (или аналог) обновлён
- [ ] Нет ссылок на порт 1234 для embedding/reranker
- [ ] Нет упоминаний cosine similarity workaround для reranker
