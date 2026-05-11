"""
20 Marks Batch Question Generator
=================================
Generates 160 long-answer questions (160 API calls × 1 each) across all
5 courses × 4 blocks × 2 bloom levels × 2 citation modes × 2 iterations.

Uses asyncio + httpx for concurrent requests.
Model: ollama-gemma4-31b

Usage:
    python3 scripts/generate_20_marks.py              # fresh run
    python3 scripts/generate_20_marks.py --resume     # skip already-completed jobs
"""

import asyncio
import csv
import functools
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

# Force unbuffered output so progress is visible even when piped to a log file
print = functools.partial(print, flush=True)  # noqa: A001

# --- Configuration ---
API_BASE = "http://localhost:8002"
MODEL_ID = "ollama-gemma4-31b"
CONCURRENCY = 1        # max simultaneous requests (Sequential mode)
TIMEOUT_S = 1200       # per-request timeout (20 mins)
NUM_QUESTIONS = 1      # questions per API call (1 for long answers to avoid cutoff)
NUM_ROUNDS = 1         # run each combination this many times
OUTPUT_CSV = Path(__file__).parent.parent / "output" / "20_marks_questions.csv"

RESUME_MODE = "--resume" in sys.argv

# --- CSV Schema (13 columns) ---
CSV_COLUMNS = [
    # Input params (7)
    "course", "block", "bloom_level", "citation_mode",
    "question_type", "model_id", "language",
    # Core Q&A (4)
    "question_number", "question", "answer", "rubric_answer",
    # Rubric — single formatted column (1)
    "rubric",
    # Metadata (2)
    "time_taken_s", "status",
]


def get_all_jobs():
    """
    Build the full job list: 5 courses × 4 blocks × 2 bloom × 2 citation × 2 iterations = 160 jobs.
    Each job produces 1 question = 160 total rows.
    """
    # All courses and their blocks (from /course-blocks endpoint)
    course_blocks = {
        "BEGC 101_ Indian Classical Literature": [
            "BLOCK CILAPPATIKARAN",
            "BLOCK KALIDASA ABHIJNANA SHAKUNTALA",
            "BLOCK SUDRAKA MRICHCHHAKATIKA",
            "BLOCK VYAS MAHABHARATA",
        ],
        "BEGC 102_ European Classical Literature": [
            "BLOCK HOMER THE ILIAD",
            "BLOCK HORACE AND OVID",
            "BLOCK PLAUTUS POT OF GOD",
            "BLOCK SOPHOCLES OEDIPUS REX",
        ],
        "BEGC 103_ Indian Writing in English": [
            "BLOCK A TIGER FOR MALGUDI",
            "BLOCK POETRY",
            "BLOCK SHORT STORY",
            "BLOCK THE BINDING VINE",
        ],
        "BEGC 104_ British Poetry and Drama_ 14th - 17th Centuries": [
            "BLOCK CHAUCER AND SPENSER",
            "BLOCK MARLOWE DOCTOR FAUSTUS",
            "BLOCK SHAKESPEARE AND DONNE",
            "BLOCK SHAKESPEARE MACBETH",
        ],
        "BEGC 105_ American Literature": [
            "BLOCK AMERICAN POETRY AN INTRODUCTION",
            "BLOCK ARTHUR MILLER ALL MY SONS",
            "BLOCK NATHANIEL HAWTHORNE THE SCARLET LETTER",
            "BLOCK SHORT FICTION",
        ],
    }

    bloom_levels = [
        ("Level 6", "Bloom level 6: Create"),
        ("Level 5", "Bloom level 5: Evaluate"),
    ]

    citation_modes = [False, True]

    jobs = []
    for course, blocks in course_blocks.items():
        for block in blocks:
            for use_citation in citation_modes:
                for bloom_label, depth_value in bloom_levels:
                    for iteration in range(1, 3):
                        jobs.append((course, block, bloom_label, depth_value, use_citation, iteration))

    return jobs


def build_request_body(course, block, depth, use_citation):
    """Build the POST /ask request payload."""
    return {
        "model_id": MODEL_ID,
        "language": "en",
        "depth": depth,
        "subject": course,
        "chapter": block,
        "standard": "12",
        "qType": "Long Answer",
        "rubric_marks": 20,
        "num_questions": NUM_QUESTIONS,
        "block": block,
        "use_rag": True,
        "use_citation": use_citation,
        "enable_alignment": False,
        "enable_dynamic_dropoff": True,
        "enable_graph_expansion": False,
        "enable_task_keywords": True,
        "temperature": 0.7,
    }


def format_rubric(rubric_data):
    """
    Format rubric dict into a single human-readable string.

    Output format:
        [Model Answer]
        The prophecy is the central catalyst...

        [Marks]
        Understanding of prophecy - 3 marks
        Analysis of character flaw - 3 marks

        [Key Points]
        1. The prophecy reveals fate vs free will
        2. Oedipus's hubris drives the tragedy
    """
    if not rubric_data:
        return ""

    parts = []

    # Marks
    marks = rubric_data.get("marks", [])
    if marks and isinstance(marks, list):
        marks_lines = []
        for m in marks:
            if isinstance(m, dict):
                criterion = m.get("criterion", "")
                mark_val = m.get("marks", "")
                marks_lines.append(f"{criterion} - {mark_val} marks")
        if marks_lines:
            parts.append("[Marks]\n" + "\n".join(marks_lines))

    # Key Points
    key_points = rubric_data.get("key_points", [])
    if key_points and isinstance(key_points, list):
        kp_lines = [f"{i}. {kp}" for i, kp in enumerate(key_points, 1)]
        parts.append("[Key Points]\n" + "\n".join(kp_lines))

    return "\n\n".join(parts)


def parse_response(response_data, course, block, bloom_label, use_citation):
    """Parse the API response into CSV rows."""
    rows = []
    citation_mode = "on" if use_citation else "off"

    for idx, item in enumerate(response_data, 1):
        # --- Build the question text (append citation if present) ---
        question_text = (item.get("question") or "").strip()
        citation_text = (item.get("citation") or "").strip()

        if citation_text:
            question_text = f'Based on the passage: "{citation_text}" — {question_text}'

        # Always use the top-level answer directly
        answer_text = (item.get("answer") or "").strip()

        # Also capture the rubric answer separately
        rubric_data = item.get("rubric") or {}
        rubric_answer_text = (rubric_data.get("answer") or "").strip()

        # --- Format rubric as single readable string ---
        rubric_formatted = format_rubric(rubric_data)

        row = {
            # Input params
            "course": course,
            "block": block,
            "bloom_level": bloom_label,
            "citation_mode": citation_mode,
            "question_type": "Long Answer",
            "model_id": MODEL_ID,
            "language": "en",
            # Core Q&A
            "question_number": idx,
            "question": question_text,
            "answer": answer_text,
            "rubric_answer": rubric_answer_text,
            # Rubric
            "rubric": rubric_formatted,
            # Metadata
            "time_taken_s": "",  # filled in by caller
            "status": "success",
        }
        rows.append(row)

    return rows


def make_error_rows(course, block, bloom_label, use_citation, error_msg):
    """Create a placeholder row when an API call fails."""
    citation_mode = "on" if use_citation else "off"
    row = {col: "" for col in CSV_COLUMNS}
    row.update({
        "course": course,
        "block": block,
        "bloom_level": bloom_label,
        "citation_mode": citation_mode,
        "question_type": "Long Answer",
        "model_id": MODEL_ID,
        "language": "en",
        "question_number": 0,
        "time_taken_s": "",
        "status": f"failed: {error_msg}",
    })
    return [row]


def job_key(course, block, bloom_label, use_citation):
    """Create a unique key for a job, used for resume logic."""
    cit = "on" if use_citation else "off"
    return f"{course}|{block}|{bloom_label}|{cit}"


def load_completed_jobs():
    """Read existing CSV and return dict of counts of already-completed job keys."""
    completed = {}
    if not OUTPUT_CSV.exists():
        return completed

    with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "success":
                key = job_key(
                    row["course"], row["block"],
                    row["bloom_level"],
                    row["citation_mode"] == "on",
                )
                completed[key] = completed.get(key, 0) + 1

    return completed


async def process_job(client, semaphore, job_num, total, course, block,
                      bloom_label, depth, use_citation, writer, csv_file):
    """Process a single API call with concurrency control."""
    citation_str = "cit:on" if use_citation else "cit:off"
    course_short = course.split("_")[0].strip()
    job_label = f"{job_num}/{total} — {course_short} / {block} / {bloom_label} / {citation_str}"

    async with semaphore:
        print(f"⏳ Starting  {job_label}")
        start = time.time()

        body = build_request_body(course, block, depth, use_citation)

        try:
            response = await client.post(
                f"{API_BASE}/ask", json=body, timeout=TIMEOUT_S
            )
            elapsed_s = round(time.time() - start, 1)

            if response.status_code != 200:
                error_msg = response.text[:100]
                print(f"   ✗ {job_label} — HTTP {response.status_code}: {error_msg} ({elapsed_s}s)")
                rows = make_error_rows(course, block, bloom_label, use_citation,
                                       f"HTTP {response.status_code}")
            else:
                data = response.json()
                rows = parse_response(data, course, block, bloom_label, use_citation)
                q_count = len(rows)
                print(f"   ✓ {job_label} — {q_count} questions ({elapsed_s}s)")

        except httpx.TimeoutException:
            elapsed_s = round(time.time() - start, 1)
            print(f"   ✗ {job_label} — Timeout after {TIMEOUT_S}s")
            rows = make_error_rows(course, block, bloom_label, use_citation, "timeout")

        except Exception as e:
            elapsed_s = round(time.time() - start, 1)
            print(f"   ✗ {job_label} — Error: {e}")
            rows = make_error_rows(course, block, bloom_label, use_citation, str(e)[:80])

        # Stamp time on every row from this call
        for row in rows:
            row["time_taken_s"] = elapsed_s

        # Write rows to CSV immediately (incremental save)
        for row in rows:
            writer.writerow(row)
        csv_file.flush()

        return rows


async def main():
    all_jobs = get_all_jobs()

    # --- Resume logic ---
    completed_counts = {}
    if RESUME_MODE:
        completed_counts = load_completed_jobs()
        if completed_counts:
            print(f"📋 Resume mode: Found {sum(completed_counts.values())} previously successful runs.")

    # Filter out completed jobs
    pending_jobs = []
    for j in all_jobs:
        course, block, bloom_label, depth, citation, iteration = j
        key = job_key(course, block, bloom_label, citation)
        if completed_counts.get(key, 0) >= iteration:
            continue
        pending_jobs.append(j)

    print("=" * 65)
    print("20 Marks Batch Question Generator")
    print(f"Total jobs:   {len(all_jobs)} API calls → {len(all_jobs)} questions")
    print(f"Pending jobs: {len(pending_jobs)} API calls → {len(pending_jobs)} questions")
    print(f"Model:        {MODEL_ID}")
    print(f"Concurrency:  {CONCURRENCY}")
    print(f"Timeout:      {TIMEOUT_S}s per request")
    print(f"Output:       {OUTPUT_CSV}")
    print(f"Resume:       {'ON' if RESUME_MODE else 'OFF'}")
    print("=" * 65)

    if not pending_jobs:
        print("\n✅ All jobs already completed! Nothing to do.")
        return

    # --- Health check ---
    async with httpx.AsyncClient() as client:
        try:
            health = await client.get(f"{API_BASE}/health", timeout=10)
            h = health.json()
            print(f"\n🟢 Server healthy — provider: {h['provider']}, docs: {h['documents_loaded']}")
        except Exception as e:
            print(f"\n🔴 Server not reachable: {e}")
            print("   Start the server first:")
            print("   cd /data1/karmela/edu_rag/backend && /data1/karmela/edu_rag/venv/bin/python app.py")
            return

    # --- Run all jobs ---
    total_start = time.time()

    # Open in append mode if resuming, write mode if fresh
    file_mode = "a" if RESUME_MODE and OUTPUT_CSV.exists() else "w"

    with open(OUTPUT_CSV, file_mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)

        # Write header only if starting fresh
        if file_mode == "w":
            writer.writeheader()

        semaphore = asyncio.Semaphore(CONCURRENCY)

        async with httpx.AsyncClient() as client:
            tasks = []
            for i, (course, block, bloom_label, depth, citation, round_num) in enumerate(pending_jobs, 1):
                task = process_job(
                    client, semaphore, i, len(pending_jobs),
                    course, block, bloom_label, depth, citation,
                    writer, csv_file,
                )
                tasks.append(task)

            await asyncio.gather(*tasks, return_exceptions=True)

    total_elapsed = round(time.time() - total_start, 1)

    # --- Summary ---
    print("\n" + "=" * 65)
    print(f"✅ Done! Completed in {total_elapsed}s ({round(total_elapsed/60, 1)} min)")
    print(f"📄 CSV saved to: {OUTPUT_CSV}")

    with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        success = sum(1 for r in rows if r["status"] == "success")
        failed = sum(1 for r in rows if r["status"].startswith("failed"))
        print(f"   ✓ {success} questions generated successfully")
        if failed:
            print(f"   ✗ {failed} rows failed (rerun with --resume to retry)")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
