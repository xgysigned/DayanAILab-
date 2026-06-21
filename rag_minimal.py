#!/usr/bin/env python
"""
Minimal RAG demo for Chapter 08.

What this verifies:
- document chunking and overlap
- embedding generation
- cosine similarity retrieval
- Top-K behavior
- lightweight reranking
- permission filtering
- prompt assembly
- online LLM call
- citations from metadata
- no-answer fallback
- simple evaluation over test questions

The online API shape is OpenAI-compatible:
- POST {EMBEDDING_API_BASE}/embeddings
- POST {LLM_API_BASE}/chat/completions
- POST {RERANK_API_BASE}/rerank

For local pipeline checks without credentials, set:
- RAG_DEMO_OFFLINE_EMBEDDINGS=1
- RAG_DEMO_OFFLINE_LLM=1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import textwrap
from datetime import datetime
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


@dataclass
class Chunk:
    chunk_id: str
    source: str
    section: str
    permission: str
    text: str


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(cleaned):
            break
        start = end - overlap
    return chunks


def chunk_docs(docs: list[dict[str, Any]], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc_index, doc in enumerate(docs):
        pieces = chunk_text(doc["text"], chunk_size, overlap)
        for piece_index, piece in enumerate(pieces):
            chunk_id = f"d{doc_index + 1:02d}_c{piece_index + 1:02d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source=doc["source"],
                    section=doc["section"],
                    permission=doc["permission"],
                    text=piece,
                )
            )
    return chunks


def post_json(url: str, api_key: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def api_url(base: str, suffix: str) -> str:
    return base.rstrip("/") + suffix


def offline_embedding(text: str, dims: int = 256) -> list[float]:
    vector = [0.0] * dims
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
    bigrams = [text[i : i + 2] for i in range(max(0, len(text) - 1))]
    for token in tokens + bigrams:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    return normalize(vector)


def embed_texts(texts: list[str]) -> list[list[float]]:
    base = os.getenv("EMBEDDING_API_BASE")
    api_key = os.getenv("EMBEDDING_API_KEY")
    model = os.getenv("EMBEDDING_MODEL")
    force_online = os.getenv("RAG_DEMO_OFFLINE_EMBEDDINGS") == "0"
    if os.getenv("RAG_DEMO_OFFLINE_EMBEDDINGS") == "1" or not (base and api_key and model):
        if force_online and not (base and api_key and model):
            raise RuntimeError(
                "Missing embedding config. Set EMBEDDING_API_BASE, EMBEDDING_API_KEY, "
                "EMBEDDING_MODEL, or set RAG_DEMO_OFFLINE_EMBEDDINGS=1."
            )
        return [offline_embedding(text) for text in texts]

    payload = {"model": model, "input": texts}
    data = post_json(api_url(base, "/embeddings"), api_key, payload)
    return [item["embedding"] for item in data["data"]]


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def accessible(required_permission: str, user_permission: str) -> bool:
    if required_permission == "employee":
        return user_permission in {"employee", "finance", "engineer", "admin"}
    if required_permission == "finance":
        return user_permission in {"finance", "admin"}
    if required_permission == "engineer":
        return user_permission in {"engineer", "admin"}
    return user_permission == required_permission or user_permission == "admin"


def retrieve(
    question: str,
    question_embedding: list[float],
    chunks: list[Chunk],
    chunk_embeddings: list[list[float]],
    top_k: int,
    user_permission: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for chunk, embedding in zip(chunks, chunk_embeddings):
        if not accessible(chunk.permission, user_permission):
            continue
        score = cosine(question_embedding, embedding)
        results.append({"chunk": chunk, "score": score, "rerank_score": score})
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def tokenize_for_rerank(text: str) -> set[str]:
    return set(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))


def rerank(question: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    q_tokens = tokenize_for_rerank(question)
    for item in candidates:
        chunk = item["chunk"]
        c_tokens = tokenize_for_rerank(chunk.text + " " + chunk.section)
        overlap = len(q_tokens & c_tokens)
        item["rerank_score"] = item["score"] + overlap * 0.03
    return sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)


def extract_json_array(text: str) -> list[dict[str, Any]]:
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        raise ValueError(f"Rerank response did not contain a JSON array: {text}")
    data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise ValueError("Rerank response JSON is not a list")
    return data


def call_chat_model(prompt: str, temperature: float = 0.0) -> str:
    base = os.getenv("LLM_API_BASE")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    if not base or not api_key or not model:
        raise RuntimeError(
            "Missing LLM config. Set LLM_API_BASE, LLM_API_KEY, LLM_MODEL, "
            "or set RAG_DEMO_OFFLINE_LLM=1."
        )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    data = post_json(api_url(base, "/chat/completions"), api_key, payload)
    return data["choices"][0]["message"]["content"]


def rerank_api_config() -> tuple[str, str, str]:
    base = os.getenv("RERANK_API_BASE") or os.getenv("LLM_API_BASE")
    api_key = os.getenv("RERANK_API_KEY") or os.getenv("LLM_API_KEY")
    model = os.getenv("RERANK_MODEL") or "BAAI/bge-reranker-v2-m3"
    if not base or not api_key or not model:
        raise RuntimeError(
            "Missing rerank config. Set RERANK_API_BASE/RERANK_API_KEY/RERANK_MODEL, "
            "or reuse LLM_API_BASE/LLM_API_KEY with RERANK_MODEL."
        )
    return base, api_key, model


def parse_rerank_results(data: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, float]:
    results = data.get("results") or data.get("data") or []
    scores: dict[str, float] = {}
    for result in results:
        index = result.get("index")
        if index is None and isinstance(result.get("document"), dict):
            index = result["document"].get("index")
        if index is None:
            chunk_id = result.get("chunk_id") or result.get("id")
        else:
            chunk_id = candidates[int(index)]["chunk"].chunk_id
        if not chunk_id:
            continue
        score = (
            result.get("relevance_score")
            if result.get("relevance_score") is not None
            else result.get("score", result.get("similarity", 0.0))
        )
        scores[str(chunk_id)] = float(score)
    return scores


def api_rerank(question: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if os.getenv("RAG_DEMO_OFFLINE_LLM") == "1":
        return rerank(question, candidates)

    base, api_key, model = rerank_api_config()
    documents = [item["chunk"].text for item in candidates]
    payload = {
        "model": model,
        "query": question,
        "documents": documents,
        "top_n": len(documents),
        "return_documents": False,
    }
    data = post_json(api_url(base, "/rerank"), api_key, payload)
    score_by_id = parse_rerank_results(data, candidates)
    for item in candidates:
        chunk: Chunk = item["chunk"]
        model_score = score_by_id.get(chunk.chunk_id, 0.0)
        item["rerank_score"] = item["score"] * 0.2 + model_score * 0.8
    return sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)


def llm_rerank(question: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if os.getenv("RAG_DEMO_OFFLINE_LLM") == "1":
        return rerank(question, candidates)

    candidate_lines = []
    for item in candidates:
        chunk: Chunk = item["chunk"]
        candidate_lines.append(
            "\n".join(
                [
                    f"chunk_id: {chunk.chunk_id}",
                    f"source: {chunk.source}",
                    f"section: {chunk.section}",
                    f"text: {chunk.text}",
                ]
            )
        )

    prompt = f"""你是 RAG 重排序模型。

请根据用户问题，给每个候选片段打相关性分数，分数范围 0 到 100。
分数越高，表示该片段越能直接回答用户问题。

要求：
1. 只输出 JSON 数组，不要输出额外解释。
2. 每个元素包含 chunk_id 和 score。
3. 不要编造不存在的 chunk_id。

用户问题：
{question}

候选片段：
{chr(10).join(candidate_lines)}

输出示例：
[
  {{"chunk_id": "d01_c01", "score": 95}},
  {{"chunk_id": "d02_c01", "score": 20}}
]
"""
    response = call_chat_model(prompt, temperature=0.0)
    scores = extract_json_array(response)
    score_by_id = {str(item["chunk_id"]): float(item["score"]) for item in scores}
    for item in candidates:
        chunk: Chunk = item["chunk"]
        model_score = score_by_id.get(chunk.chunk_id, 0.0) / 100.0
        item["rerank_score"] = item["score"] * 0.2 + model_score * 0.8
    return sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)


def rerank_candidates(question: str, candidates: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "none":
        return candidates
    if mode == "rule":
        return rerank(question, candidates)
    if mode == "api":
        return api_rerank(question, candidates)
    if mode == "llm":
        return llm_rerank(question, candidates)
    raise ValueError(f"Unsupported rerank mode: {mode}")


def format_context(items: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in items:
        chunk: Chunk = item["chunk"]
        parts.append(
            "\n".join(
                [
                    f"[{chunk.chunk_id}]",
                    f"source: {chunk.source}",
                    f"section: {chunk.section}",
                    f"text: {chunk.text}",
                ]
            )
        )
    return "\n\n".join(parts)


def build_prompt(question: str, context_items: list[dict[str, Any]]) -> str:
    context = format_context(context_items)
    return f"""你是企业知识库问答助手。

请只根据下面提供的资料回答用户问题。
如果资料中没有答案，请回答“资料中未提到”。
不要编造资料中没有的信息。
回答必须包含引用来源，引用只能使用资料中的 chunk_id、source 和 section。

资料：
{context}

用户问题：
{question}

输出格式：
答案：...
引用：chunk_id / source / section
"""


def offline_llm_answer(question: str, context_items: list[dict[str, Any]]) -> str:
    joined = "\n".join(item["chunk"].text for item in context_items)
    if ("十年" in question or "10 年" in question or "10年" in question) and "10 天" in joined:
        first = context_items[0]["chunk"]
        return f"答案：员工连续工作满十年后可享受 10 天带薪年假。\n引用：{first.chunk_id} / {first.source} / {first.section}"
    if "年假" in question and "5 天" in joined:
        first = context_items[0]["chunk"]
        return f"答案：员工入职满一年后可享受 5 天带薪年假。\n引用：{first.chunk_id} / {first.source} / {first.section}"
    if "导出" in question and "导出权限" in joined:
        first = context_items[0]["chunk"]
        return f"答案：导出报表失败时，请先确认当前账号是否拥有导出权限，并检查筛选条件是否过大。\n引用：{first.chunk_id} / {first.source} / {first.section}"
    if "报销" in question and "部门负责人和财务负责人" in joined:
        first = context_items[0]["chunk"]
        return f"答案：单笔金额超过 5000 元的报销，需要部门负责人和财务负责人共同审批。\n引用：{first.chunk_id} / {first.source} / {first.section}"
    if ("登录" in question or "密码" in question or "输错" in question or "锁定" in question) and "30 分钟" in joined:
        first = context_items[0]["chunk"]
        return f"答案：连续输错 5 次密码后，账号会被临时锁定 30 分钟。\n引用：{first.chunk_id} / {first.source} / {first.section}"
    if "接口" in question and "上游服务状态" in joined:
        first = context_items[0]["chunk"]
        return f"答案：接口超时时，请先检查上游服务状态、数据库连接池使用率和网关超时配置。\n引用：{first.chunk_id} / {first.source} / {first.section}"
    return "答案：资料中未提到。\n引用：无"


def call_llm(prompt: str, question: str, context_items: list[dict[str, Any]]) -> str:
    has_online_llm = bool(os.getenv("LLM_API_BASE") and os.getenv("LLM_API_KEY") and os.getenv("LLM_MODEL"))
    force_online = os.getenv("RAG_DEMO_OFFLINE_LLM") == "0"
    if os.getenv("RAG_DEMO_OFFLINE_LLM") == "1" or not has_online_llm:
        if force_online and not has_online_llm:
            raise RuntimeError(
                "Missing LLM config. Set LLM_API_BASE, LLM_API_KEY, LLM_MODEL, "
                "or set RAG_DEMO_OFFLINE_LLM=1."
            )
        return offline_llm_answer(question, context_items)
    return call_chat_model(prompt, temperature=0.2)


def print_candidates(title: str, items: list[dict[str, Any]]) -> None:
    print(f"\n{title}")
    for rank, item in enumerate(items, start=1):
        chunk: Chunk = item["chunk"]
        print(
            f"{rank}. {chunk.chunk_id} | score={item['score']:.4f} | "
            f"rerank={item['rerank_score']:.4f} | {chunk.source} / {chunk.section} / {chunk.permission}"
        )
        print(f"   {chunk.text}")


def normalize_for_eval(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def run_one(
    question: str,
    user_permission: str,
    chunks: list[Chunk],
    chunk_embeddings: list[list[float]],
    top_k: int,
    max_context: int,
    rerank_mode: str,
    verbose: bool,
) -> dict[str, Any]:
    question_embedding = embed_texts([question])[0]
    candidates = retrieve(question, question_embedding, chunks, chunk_embeddings, top_k, user_permission)
    reranked = rerank_candidates(question, candidates, rerank_mode)
    context_items = reranked[:max_context]
    prompt = build_prompt(question, context_items)
    answer = call_llm(prompt, question, context_items)

    if verbose:
        print(f"\nQUESTION: {question}")
        print(f"USER_PERMISSION: {user_permission}")
        print(f"QUERY_EMBEDDING_DIM: {len(question_embedding)}")
        print(f"RERANK_MODE: {rerank_mode}")
        print_candidates("TOP-K CANDIDATES", candidates)
        print_candidates("RERANKED CONTEXT", context_items)
        print("\nPROMPT")
        print("-" * 80)
        print(prompt)
        print("-" * 80)
        print("\nANSWER")
        print(answer)

    return {
        "question": question,
        "user_permission": user_permission,
        "answer": answer,
        "candidates": candidates,
        "context_items": context_items,
    }


def evaluate(args: argparse.Namespace, chunks: list[Chunk], chunk_embeddings: list[list[float]]) -> None:
    questions = load_json(ROOT / "eval_questions.json")
    passed = 0
    total = len(questions)
    report_cases: list[dict[str, Any]] = []
    print(f"\nRunning evaluation: {total} cases")
    for case in questions:
        result = run_one(
            case["question"],
            case["user_permission"],
            chunks,
            chunk_embeddings,
            args.top_k,
            args.max_context,
            args.rerank_mode,
            verbose=False,
        )
        answer = result["answer"]
        context_sources = {item["chunk"].source for item in result["context_items"]}
        contains_ok = normalize_for_eval(case["expected_contains"]) in normalize_for_eval(answer)
        source_ok = case["expected_source"] is None or case["expected_source"] in context_sources
        passed_case = contains_ok and source_ok
        passed += 1 if passed_case else 0
        status = "PASS" if passed_case else "FAIL"
        print(f"\n[{status}] {case['question']} ({case['user_permission']})")
        print(f"expected_contains: {case['expected_contains']!r} -> {contains_ok}")
        print(f"expected_source: {case['expected_source']!r} -> {source_ok}")
        print(f"answer: {answer}")
        print("context:", ", ".join(sorted(context_sources)) or "none")
        report_cases.append(
            {
                "status": status,
                "question": case["question"],
                "permission": case["user_permission"],
                "expected_contains": case["expected_contains"],
                "contains_ok": contains_ok,
                "expected_source": case["expected_source"],
                "source_ok": source_ok,
                "answer": answer,
                "context_sources": sorted(context_sources),
                "context_chunks": [
                    {
                        "chunk_id": item["chunk"].chunk_id,
                        "source": item["chunk"].source,
                        "section": item["chunk"].section,
                        "score": item["score"],
                        "rerank_score": item["rerank_score"],
                    }
                    for item in result["context_items"]
                ],
            }
        )
    print(f"\nEVAL SUMMARY: {passed}/{total} passed")
    if args.report:
        write_report(args.report, args, passed, total, report_cases)


def write_report(path: str, args: argparse.Namespace, passed: int, total: int, cases: list[dict[str, Any]]) -> None:
    target = Path(path)
    if not target.is_absolute():
        target = ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RAG Demo 测试报告",
        "",
        f"- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Embedding 模式：{'online' if has_online_embedding_config() and os.getenv('RAG_DEMO_OFFLINE_EMBEDDINGS') != '1' else 'offline'}",
        f"- LLM 模式：{'online' if has_online_llm_config() and os.getenv('RAG_DEMO_OFFLINE_LLM') != '1' else 'offline'}",
        f"- 重排模式：{args.rerank_mode}",
        f"- Rerank 模型：{os.getenv('RERANK_MODEL') or 'N/A'}",
        f"- chunk_size：{args.chunk_size}",
        f"- overlap：{args.overlap}",
        f"- top_k：{args.top_k}",
        f"- max_context：{args.max_context}",
        f"- 结果：{passed}/{total} passed",
        "",
        "## 覆盖功能点",
        "",
        "- 文档切分与 overlap",
        "- 在线 Embedding",
        "- 内存向量检索与 Top-K",
        "- 专用 reranker 重排",
        "- 权限过滤",
        "- Prompt 拼接",
        "- 在线 LLM 生成",
        "- 引用来源",
        "- 无答案兜底",
        "- 评估归一化",
        "",
        "## 用例结果",
        "",
        "| 状态 | 问题 | 权限 | 期望片段 | 期望内容 | 检索来源 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in cases:
        lines.append(
            f"| {case['status']} | {case['question']} | {case['permission']} | "
            f"{case['expected_source'] or '无'} | {case['expected_contains']} | "
            f"{', '.join(case['context_sources']) or '无'} |"
        )
    lines.extend(["", "## 详细输出", ""])
    for case in cases:
        lines.extend(
            [
                f"### {case['status']} - {case['question']}",
                "",
                f"- 权限：`{case['permission']}`",
                f"- 期望内容命中：`{case['contains_ok']}`",
                f"- 期望来源命中：`{case['source_ok']}`",
                "",
                "检索上下文：",
                "",
            ]
        )
        for chunk in case["context_chunks"]:
            lines.append(
                f"- `{chunk['chunk_id']}` {chunk['source']} / {chunk['section']} "
                f"(score={chunk['score']:.4f}, rerank={chunk['rerank_score']:.4f})"
            )
        lines.extend(["", "答案：", "", "```text", case["answer"].strip(), "```", ""])
    target.write_text("\n".join(lines), encoding="utf-8")
    print(f"REPORT WRITTEN: {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal RAG demo")
    parser.add_argument("--question", default="入职一年后年假有几天？")
    parser.add_argument("--user-permission", default="employee", choices=["employee", "finance", "engineer", "admin"])
    parser.add_argument("--chunk-size", type=int, default=80)
    parser.add_argument("--overlap", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context", type=int, default=3)
    parser.add_argument(
        "--rerank-mode",
        default=os.getenv("RAG_DEMO_RERANK_MODE", "rule"),
        choices=["none", "rule", "api", "llm"],
        help="none: keep vector order, rule: local lexical rerank, api: rerank API, llm: chat model scores candidates",
    )
    parser.add_argument("--eval", action="store_true", help="Run eval_questions.json")
    parser.add_argument("--report", default="", help="Write evaluation report to a markdown file")
    parser.add_argument("--require-online", action="store_true", help="Fail if online embedding or LLM config is missing")
    parser.add_argument("--show-chunks", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def has_online_embedding_config() -> bool:
    return bool(os.getenv("EMBEDDING_API_BASE") and os.getenv("EMBEDDING_API_KEY") and os.getenv("EMBEDDING_MODEL"))


def has_online_llm_config() -> bool:
    return bool(os.getenv("LLM_API_BASE") and os.getenv("LLM_API_KEY") and os.getenv("LLM_MODEL"))


def has_online_rerank_config() -> bool:
    return bool((os.getenv("RERANK_API_BASE") or os.getenv("LLM_API_BASE")) and (os.getenv("RERANK_API_KEY") or os.getenv("LLM_API_KEY")) and (os.getenv("RERANK_MODEL") or "BAAI/bge-reranker-v2-m3"))


def print_runtime_modes(rerank_mode: str) -> None:
    embedding_mode = "online" if has_online_embedding_config() and os.getenv("RAG_DEMO_OFFLINE_EMBEDDINGS") != "1" else "offline"
    llm_mode = "online" if has_online_llm_config() and os.getenv("RAG_DEMO_OFFLINE_LLM") != "1" else "offline"
    rerank_detail = "api-online" if rerank_mode == "api" and has_online_rerank_config() else rerank_mode
    print(f"RUNTIME MODES: embeddings={embedding_mode}, llm={llm_mode}, rerank={rerank_detail}")


def main() -> int:
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / ".env.local")
    if not has_online_embedding_config() and not has_online_llm_config():
        load_dotenv(ROOT / ".env.example")
    args = parse_args()

    if args.require_online and (not has_online_embedding_config() or not has_online_llm_config()):
        raise RuntimeError(
            "Online mode required, but .env/config is incomplete. Need EMBEDDING_API_BASE, "
            "EMBEDDING_API_KEY, EMBEDDING_MODEL, LLM_API_BASE, LLM_API_KEY, and LLM_MODEL."
        )
    if not args.quiet:
        print_runtime_modes(args.rerank_mode)

    docs = load_json(ROOT / "knowledge_base.json")
    chunks = chunk_docs(docs, args.chunk_size, args.overlap)
    chunk_embeddings = embed_texts([chunk.text for chunk in chunks])

    if args.show_chunks:
        print(f"CHUNKS ({len(chunks)})")
        for chunk in chunks:
            print(f"- {chunk.chunk_id} | {chunk.source} / {chunk.section} / {chunk.permission}")
            print(textwrap.indent(chunk.text, "  "))

    if args.eval:
        evaluate(args, chunks, chunk_embeddings)
    else:
        run_one(
            args.question,
            args.user_permission,
            chunks,
            chunk_embeddings,
            args.top_k,
            args.max_context,
            args.rerank_mode,
            verbose=not args.quiet,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
