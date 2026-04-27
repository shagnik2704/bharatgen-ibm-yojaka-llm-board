"""Inspect generation prompt size and estimated token usage for /ask requests.

Usage examples:
  python inspect_prompt.py --subject "BEGC 101_ Indian Classical Literature" --chapter "Block 1" --qtype "Short Answer"
  python inspect_prompt.py --subject "..." --chapter "..." --qtype "True/False" --without-retrieval
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from .prompt_builder import build_prompt_from_request
    from .rag_retriever import MinimalRAGRetriever, DEFAULT_K
    from .model_runner import get_completion_token_budget
except ImportError:
    from prompt_builder import build_prompt_from_request
    from rag_retriever import MinimalRAGRetriever, DEFAULT_K
    from model_runner import get_completion_token_budget


@dataclass
class PromptReq:
    model_id: str
    language: str
    depth: str
    subject: str
    chapter: str
    standard: str
    theme: str
    qType: str
    num_questions: int
    block: Optional[str] = None


def estimate_tokens_simple(text: str) -> int:
    # Rough multilingual estimate for BPE families.
    # Better than words for punctuation-heavy prompts.
    return max(1, int(len(text) / 4))


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def summarize_text(text: str) -> Dict[str, Any]:
    return {
        "chars": len(text or ""),
        "words": count_words(text or ""),
        "est_tokens": estimate_tokens_simple(text or ""),
    }


def print_summary(label: str, stats: Dict[str, Any]) -> None:
    print(f"{label}: chars={stats['chars']}, words={stats['words']}, est_tokens={stats['est_tokens']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect prompt size and estimated Groq token usage.")
    parser.add_argument("--model-id", default="groq-llama-70b")
    parser.add_argument("--language", default="en")
    parser.add_argument("--depth", default="Bloom level 1: Remember")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--chapter", required=True)
    parser.add_argument("--standard", default="")
    parser.add_argument("--theme", default="general")
    parser.add_argument("--qtype", default="Short Answer")
    parser.add_argument("--num-questions", type=int, default=1)
    parser.add_argument("--block", default=None)
    parser.add_argument("--without-retrieval", action="store_true")
    parser.add_argument(
        "--rag-store-dir",
        default=str((Path(__file__).resolve().parent.parent / "rag_store_books").resolve()),
        help="Path to rag_store_books",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args()

    req = PromptReq(
        model_id=args.model_id,
        language=args.language,
        depth=args.depth,
        subject=args.subject,
        chapter=args.chapter,
        standard=args.standard,
        theme=args.theme,
        qType=args.qtype,
        num_questions=max(1, int(args.num_questions)),
        block=args.block,
    )

    retrieval_query = " ".join([req.chapter or "", req.theme or ""]).strip() or req.subject
    chunk_text = ""
    top_meta = {}

    if not args.without_retrieval:
        retriever = MinimalRAGRetriever(Path(args.rag_store_dir))
        chunk_text, metas = retriever.retrieve(
            query=retrieval_query,
            subject=req.subject,
            chapter=req.chapter,
            standard=req.standard,
            block=req.block,
            k=max(1, int(args.k)),
        )
        top_meta = metas[0] if metas else {}

    template_prompt = build_prompt_from_request(req, "")
    full_prompt = build_prompt_from_request(req, chunk_text)

    print("=== Prompt Inspection ===")
    print(f"model_id={req.model_id}")
    print(f"retrieval_enabled={not args.without_retrieval}")
    print(f"retrieval_query={retrieval_query}")
    print()

    print_summary("template_prompt", summarize_text(template_prompt))
    print_summary("retrieved_chunk", summarize_text(chunk_text))
    print_summary("full_prompt", summarize_text(full_prompt))

    max_output_tokens = get_completion_token_budget(req.model_id, req=req)
    est_requested_tokens = summarize_text(full_prompt)["est_tokens"] + int(max_output_tokens)
    legacy_hard_caps = {
        "groq-llama-8b": 65536,
        "groq-llama-70b": 32768,
        "groq-qwen-32b": 32768,
        "rag-piped-groq-70b": 32768,
        "groq-llama-guard": 1024,
        "groq-gpt-oss-120b": 65536,
        "groq-gpt-oss-20b": 65536,
    }
    legacy_cap = legacy_hard_caps.get(req.model_id)
    legacy_est_requested = (
        summarize_text(full_prompt)["est_tokens"] + int(legacy_cap)
        if legacy_cap is not None
        else None
    )

    print()
    print(f"max_output_tokens_budget={max_output_tokens}")
    print(f"est_requested_tokens=input+output={est_requested_tokens}")
    if legacy_est_requested is not None:
        print(f"legacy_hard_cap_output_tokens={legacy_cap}")
        print(f"legacy_est_requested_tokens=input+output={legacy_est_requested}")

    duplicate_source_header = full_prompt.count("### SOURCE MATERIAL")
    duplicate_output_header = full_prompt.count("### OUTPUT FORMAT")
    duplicate_chunk_occurrences = full_prompt.count(chunk_text) if chunk_text else 0

    print()
    print("=== Redundancy Checks ===")
    print(f"count('### SOURCE MATERIAL')={duplicate_source_header}")
    print(f"count('### OUTPUT FORMAT')={duplicate_output_header}")
    print(f"chunk_occurrences_in_prompt={duplicate_chunk_occurrences}")

    if top_meta:
        print()
        print("=== Retrieved Top Chunk Meta ===")
        print(json.dumps(top_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
