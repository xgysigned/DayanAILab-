# RAG Minimal Demo

这个目录用于验证第八章 RAG 的核心知识点：

- 文档切分
- Embedding
- 向量检索
- Top-K
- 重排序：默认本地规则重排，可选专用 reranker 重排
- Prompt 拼接
- 线上模型生成
- 引用来源
- 无答案兜底
- 权限过滤
- 测试集评估

## 当前向量库实现

当前 demo 使用的是**内存向量库**：

```text
chunks 列表保存文档片段
chunk_embeddings 列表保存向量
检索时遍历所有向量并计算余弦相似度
排序后取 Top-K
```

这样做是为了教学验证：

- 不需要安装数据库。
- 不需要启动额外服务。
- 能直接看到 chunk、embedding、Top-K、重排序、Prompt 和答案。
- 方便验证本章所有核心概念。

它不是生产级方案。后续可以把 `retrieve()` 和向量存储部分替换为：

| 向量库 | 适合场景 |
| --- | --- |
| FAISS | 本地高性能向量检索 |
| PostgreSQL + pgvector | 业务系统集成、元数据过滤、权限控制 |
| Qdrant | 独立向量库服务，开发体验好 |
| Milvus | 更大规模向量检索 |
| Elasticsearch / OpenSearch | 关键词 + 向量混合检索 |

替换时核心链路不变：

```text
文档切分 -> Embedding -> 向量入库 -> 查询向量 -> Top-K 检索 -> 重排序 -> Prompt -> 生成答案
```

## 当前重排序实现

当前 demo 有 3 种重排序模式：

| 模式 | 命令参数 | 说明 |
| --- | --- | --- |
| 不重排 | `--rerank-mode none` | 保持向量检索 Top-K 顺序 |
| 本地规则重排 | `--rerank-mode rule` | 默认模式，用词项重合对候选片段轻量加分 |
| 专用 reranker 重排 | `--rerank-mode api` | 调用 rerank API，例如 `BAAI/bge-reranker-v2-m3` |
| 聊天模型模拟重排 | `--rerank-mode llm` | 调用线上聊天模型，让模型给候选片段打 0-100 相关性分 |

严格来说，`rule` 不是重排模型，只是为了离线教学验证链路。  
`api` 模式是真正的重排模型调用，当前使用 `BAAI/bge-reranker-v2-m3`。`llm` 模式只是聊天模型模拟重排，用于没有 rerank API 时临时验证。

使用专用 reranker 重排：

```powershell
python rag_minimal.py --question "工作十年后年假有几天？" --rerank-mode api --require-online
```

如果要接不同格式的 rerank API，可以改 `rag_minimal.py` 里的 `api_rerank()`。

## 推荐运行命令

离线完整验证：

```powershell
python rag_minimal.py --eval
```

线上完整验证，使用专用 reranker：

```powershell
python rag_minimal.py --eval --rerank-mode api --require-online
```

生成可追溯报告：

```powershell
python rag_minimal.py --eval --rerank-mode api --require-online --report reports/rag-online-rerank-report.md
```

单条问题调试：

```powershell
python rag_minimal.py --question "工作十年后年假有几天？" --user-permission employee --rerank-mode api --require-online
```

运行时重点观察：

- `RUNTIME MODES` 是否显示 `embeddings=online, llm=online, rerank=api-online`
- `TOP-K CANDIDATES` 是否召回正确片段
- `RERANKED CONTEXT` 是否把正确片段排前面
- `PROMPT` 是否只包含有权限的资料
- `ANSWER` 是否包含答案和引用

## 文件说明

```text
rag_demo/
  rag_minimal.py        # 最小 RAG 主程序
  knowledge_base.json   # 模拟企业知识库
  eval_questions.json   # 测试问题
  .env.example          # 线上模型配置模板
```

## 线上模型配置

复制 `.env.example` 为 `.env`，然后填入你提供的线上模型配置：

```text
LLM_API_BASE=https://your-model-api.example.com/v1
LLM_API_KEY=your_api_key
LLM_MODEL=your_chat_model

EMBEDDING_API_BASE=https://your-embedding-api.example.com/v1
EMBEDDING_API_KEY=your_api_key
EMBEDDING_MODEL=your_embedding_model

RERANK_API_BASE=https://your-rerank-api.example.com/v1
RERANK_API_KEY=your_api_key
RERANK_MODEL=BAAI/bge-reranker-v2-m3
```

接口默认按 OpenAI-compatible 形式调用：

- `POST {EMBEDDING_API_BASE}/embeddings`
- `POST {LLM_API_BASE}/chat/completions`
- `POST {RERANK_API_BASE}/rerank`

如果你的接口不是这个格式，只需要改 `rag_minimal.py` 里的：

- `embed_texts()`
- `call_llm()`

确认线上配置是否生效：

```powershell
python rag_minimal.py --eval --require-online
```

运行时会打印：

```text
RUNTIME MODES: embeddings=online, llm=online, rerank=rule
```

如果显示 `offline`，说明没有读到 `.env`，或者 `.env` 缺少必要配置。

专用 reranker 重排：

```powershell
python rag_minimal.py --eval --rerank-mode api --require-online
```

本次线上测试报告：

```text
reports/rag-online-rerank-report.md
```

## 离线验证

没有线上模型时，脚本会自动使用离线 fallback，验证 RAG 链路。你可以直接运行：

```powershell
python rag_minimal.py --eval
```

也可以显式设置离线模式：

PowerShell:

```powershell
$env:RAG_DEMO_OFFLINE_EMBEDDINGS="1"
$env:RAG_DEMO_OFFLINE_LLM="1"
python rag_minimal.py --show-chunks
```

运行单个问题：

```powershell
python rag_minimal.py --question "入职一年后年假有几天？" --user-permission employee
```

运行评估集：

```powershell
python rag_minimal.py --eval
```

## 验证权限过滤

普通员工不能看到财务制度：

```powershell
python rag_minimal.py --question "单笔 8000 元报销需要谁审批？" --user-permission employee
```

财务人员可以看到财务制度：

```powershell
python rag_minimal.py --question "单笔 8000 元报销需要谁审批？" --user-permission finance
```

## 验证 Top-K 和切分

调整 Top-K：

```powershell
python rag_minimal.py --question "工作十年后年假有几天？" --top-k 3
python rag_minimal.py --question "工作十年后年假有几天？" --top-k 8
```

调整切分：

```powershell
python rag_minimal.py --chunk-size 40 --overlap 10 --show-chunks
python rag_minimal.py --chunk-size 120 --overlap 30 --show-chunks
```

观察点：

- chunk 是否保留完整语义
- Top-K 是否召回正确片段
- 重排序后正确片段是否靠前
- Prompt 是否只包含用户有权限的资料
- 答案是否包含引用
- 无答案时是否回答“资料中未提到”

## 本 demo 的边界

这是教学用最小 demo，不是生产级 RAG 系统。生产系统还需要：

- 更强的文档解析
- 更可靠的向量数据库
- 更好的重排序模型
- 更严格的引用校验
- 更完整的权限体系
- 更系统的评估和监控
