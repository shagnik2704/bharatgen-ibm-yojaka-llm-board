"""
Trial Batch Question Generator
================================
Generates 20 questions (10 API calls × 2 each) across representative
course/block/bloom/citation combinations. Uses asyncio for concurrency.
"""

import asyncio
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import httpx

# --- Configuration ---
API_BASE = "http://localhost:8002"
MODEL_ID = "ollama-gemma4-e4b"
CONCURRENCY = 3  # max simultaneous requests
TIMEOUT_S = 180  # per-request timeout
OUTPUT_CSV = Path(__file__).parent.parent / "output" / "trial_questions.csv"

# --- Trial Jobs (10 calls → 20 questions) ---
JOBS = [
    # (course, block, bloom_level_label, depth_value, citation)
    ("BEGC 101_ Indian Classical Literature", "BLOCK CILAPPATIKARAN", "Level 1", "Bloom level 1: Remember", False),
    ("BEGC 101_ Indian Classical Literature", "BLOCK CILAPPATIKARAN", "Level 1", "Bloom level 1: Remember", True),
    ("BEGC 102_ European Classical Literature", "BLOCK SOPHOCLES OEDIPUS REX", "Level 1", "Bloom level 1: Remember", False),
    ("BEGC 102_ European Classical Literature", "BLOCK SOPHOCLES OEDIPUS REX", "Level 2", "Bloom level 2: Understand", False),
    ("BEGC 103_ Indian Writing in English", "BLOCK POETRY", "Level 2", "Bloom level 2: Understand", True),
    ("BEGC 103_ Indian Writing in English", "BLOCK POETRY", "Level 1", "Bloom level 1: Remember", False),
    ("BEGC 104_ British Poetry and Drama_ 14th - 17th Centuries", "BLOCK SHAKESPEARE MACBETH", "Level 1", "Bloom level 1: Remember", True),
    ("BEGC 104_ British Poetry and Drama_ 14th - 17th Centuries", "BLOCK SHAKESPEARE MACBETH", "Level 2", "Bloom level 2: Understand", False),
    ("BEGC 105_ American Literature", "BLOCK SHORT FICTION", "Level 1", "Bloom level 1: Remember", False),
    ("BEGC 105_ American Literature", "BLOCK SHORT FICTION", "Level 2", "Bloom level 2: Understand", True),
]

# --- CSV Header ---
CSV_COLUMNS = [
    # Input params
    "course", "block", "bloom_level", "citation_mode", "question_type",
    "model_id", "language",
    # Core Q&A
    "question_number", "question", "answer", "citation",
    # Rubric (Level 2 only)
    "rubric_answer", "rubric_marks", "rubric_key_points",
    # Metadata
    "generation_time_ms", "similarity_score", "source_title",
    "source_path", "type_match", "status",
]


def build_request_body(course, block, depth, use_citation):
    """Build the POST /ask request payload."""
    return {
        "model_id": MODEL_ID,
        "language": "en",
        "depth": depth,
        "subject": course,
        "chapter": block,
        "standard": "12",
        "qType": "Short Answer",
        "num_questions": 2,
        "block": block,
        "use_rag": True,
        "use_citation": use_citation,
        "enable_alignment": False,
        "enable_dynamic_dropoff": True,
        "enable_graph_expansion": False,
        "enable_task_keywords": True,
        "temperature": 0.7,
    }


def parse_response(response_data, course, block, bloom_label, use_citation):
    """Parse the API response into CSV rows."""
    rows = []
    citation_mode = "on" if use_citation else "off"

    for idx, item in enumerate(response_data, 1):
        # Extract rubric fields (Bloom Level 2 returns JSON with rubric)
        rubric = item.get("rubric") or {}
        rubric_answer = rubric.get("answer", "")
        rubric_marks = json.dumps(rubric.get("marks", []), ensure_ascii=False) if rubric.get("marks") else ""
        rubric_key_points = json.dumps(rubric.get("key_points", []), ensure_ascii=False) if rubric.get("key_points") else ""

        # Extract source metadata
        source_meta = item.get("source_meta") or {}

        row = {
            # Input params
            "course": course,
            "block": block,
            "bloom_level": bloom_label,
            "citation_mode": citation_mode,
            "question_type": "Short Answer",
            "model_id": MODEL_ID,
            "language": "en",
            # Core Q&A
            "question_number": idx,
            "question": (item.get("question") or "").strip(),
            "answer": (item.get("answer") or "").strip(),
            "citation": (item.get("citation") or "").strip(),
            # Rubric
            "rubric_answer": rubric_answer,
            "rubric_marks": rubric_marks,
            "rubric_key_points": rubric_key_points,
            # Metadata
            "generation_time_ms": item.get("generation_time_ms", ""),
            "similarity_score": source_meta.get("similarity", ""),
            "source_title": source_meta.get("title", ""),
            "source_path": source_meta.get("pdf_path", ""),
            "type_match": item.get("type_match", ""),
            "status": "success",
        }
        rows.append(row)

    return rows


def make_error_rows(course, block, bloom_label, use_citation, error_msg):
    """Create placeholder rows when an API call fails."""
    citation_mode = "on" if use_citation else "off"
    row = {col: "" for col in CSV_COLUMNS}
    row.update({
        "course": course,
        "block": block,
        "bloom_level": bloom_label,
        "citation_mode": citation_mode,
        "question_type": "Short Answer",
        "model_id": MODEL_ID,
        "language": "en",
        "question_number": 0,
        "question": "",
        "answer": "",
        "status": f"failed: {error_msg}",
    })
    return [row]


async def process_job(client, semaphore, job_num, total, course, block, bloom_label, depth, use_citation, writer, csv_file):
    """Process a single API call with concurrency control."""
    citation_str = "citation:on" if use_citation else "citation:off"
    job_label = f"{job_num}/{total} — {course[:20]}… / {block} / {bloom_label} / {citation_str}"

    async with semaphore:
        print(f"⏳ Starting {job_label}")
        start = time.time()

        body = build_request_body(course, block, depth, use_citation)

        try:
            response = await client.post(f"{API_BASE}/ask", json=body, timeout=TIMEOUT_S)
            elapsed = round((time.time() - start) * 1000, 1)

            if response.status_code != 200:
                error_msg = response.text[:100]
                print(f"  ✗ {job_label} — HTTP {response.status_code}: {error_msg} ({elapsed}ms)")
                rows = make_error_rows(course, block, bloom_label, use_citation, f"HTTP {response.status_code}")
            else:
                data = response.json()
                rows = parse_response(data, course, block, bloom_label, use_citation)
                q_count = len(rows)
                print(f"  ✓ {job_label} — {q_count} questions ({elapsed}ms)")

        except httpx.TimeoutException:
            elapsed = round((time.time() - start) * 1000, 1)
            print(f"  ✗ {job_label} — Timeout after {TIMEOUT_S}s")
            rows = make_error_rows(course, block, bloom_label, use_citation, "timeout")

        except Exception as e:
            elapsed = round((time.time() - start) * 1000, 1)
            print(f"  ✗ {job_label} — Error: {e}")
            rows = make_error_rows(course, block, bloom_label, use_citation, str(e)[:80])

        # Write rows to CSV immediately (incremental save)
        for row in rows:
            writer.writerow(row)
        csv_file.flush()

        return rows


async def main():
    print("=" * 60)
    print("Trial Batch Question Generator")
    print(f"Jobs: {len(JOBS)} API calls → ~{len(JOBS) * 2} questions")
    print(f"Model: {MODEL_ID}")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Output: {OUTPUT_CSV}")
    print("=" * 60)

    # Health check
    async with httpx.AsyncClient() as client:
        try:
            health = await client.get(f"{API_BASE}/health", timeout=10)
            h = health.json()
            print(f"\n🟢 Server healthy — provider: {h['provider']}, docs: {h['documents_loaded']}")
        except Exception as e:
            print(f"\n🔴 Server not reachable: {e}")
            print("   Start the server first: cd /data1/karmela/edu_rag/backend && python app.py")
            return

    # Open CSV and run all jobs
    total_start = time.time()
    all_rows = []

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        semaphore = asyncio.Semaphore(CONCURRENCY)

        async with httpx.AsyncClient() as client:
            tasks = []
            for i, (course, block, bloom_label, depth, citation) in enumerate(JOBS, 1):
                task = process_job(
                    client, semaphore, i, len(JOBS),
                    course, block, bloom_label, depth, citation,
                    writer, csv_file,
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

    total_elapsed = round(time.time() - total_start, 1)

    # Summary
    print("\n" + "=" * 60)
    print(f"✅ Done! Generated in {total_elapsed}s")
    print(f"📄 CSV saved to: {OUTPUT_CSV}")

    # Count results
    with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        success = sum(1 for r in rows if r["status"] == "success")
        failed = sum(1 for r in rows if r["status"].startswith("failed"))
        print(f"   ✓ {success} questions generated successfully")
        if failed:
            print(f"   ✗ {failed} rows failed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
