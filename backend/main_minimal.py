import json
import os
import re
import sqlite3
import uuid
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from .rag_retriever import MAX_CHUNK_WORDS, MIN_CHUNK_WORDS, MinimalRAGRetriever
    from .parse_ai_output import parse_ai_output
    from .GEval import GEval
    from .model_runner import run_model, set_clients
    from .council import run_council_flow
    from .prompt_builder import build_prompt_from_request, get_generation_question_count
except ImportError:
    from rag_retriever import MAX_CHUNK_WORDS, MIN_CHUNK_WORDS, MinimalRAGRetriever
    from parse_ai_output import parse_ai_output
    from GEval import GEval
    from model_runner import run_model, set_clients
    from council import run_council_flow
    from prompt_builder import build_prompt_from_request, get_generation_question_count


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
RAG_STORE_DIR = Path(os.getenv("RAG_STORE_DIR", str(PROJECT_ROOT / "rag_store_books"))).resolve()
DEFAULT_K = 5
DB_PATH = Path(os.getenv("MINIMAL_DB_PATH", str(BASE_DIR / "minimal_questions.db"))).resolve()


def _ensure_model_id(model_id: Optional[str], fallback: str) -> str:
    value = (model_id or "").strip()
    if value.startswith("groq-") or value.startswith("ollama-") or value in {"rag-piped-groq-70b"}:
        return value
    return fallback


def _normalize_legacy_generation_model_id(model_id: str) -> str:
    # Backward-compat alias: after introducing explicit use_rag, this legacy
    # model ID behaves the same as regular Groq Llama.
    if model_id == "rag-piped-groq-70b":
        return "groq-llama-70b"
    return model_id


GEVAL_MODEL = _ensure_model_id(os.getenv("GEVAL_MODEL", "ollama-gemma4-e4b"), "ollama-gemma4-e4b")
GEVAL_MODEL_2 = _ensure_model_id(os.getenv("GEVAL_MODEL_2", "ollama-phi4-mini"), "ollama-phi4-mini")

param_ncert = GEval(model=GEVAL_MODEL, groq_api_key=os.getenv("GROQ_API_KEY", "").strip(), likert_scale=[1, 2, 3, 4, 5])
llama_bloom = GEval(model=GEVAL_MODEL_2, groq_api_key=os.getenv("GROQ_API_KEY", "").strip(), likert_scale=[1, 2, 3, 4, 5])
guardrails_qwen = GEval(model=GEVAL_MODEL_2, groq_api_key=os.getenv("GROQ_API_KEY", "").strip(), likert_scale=[1, 2])
verification_llama = GEval(model=GEVAL_MODEL_2, groq_api_key=os.getenv("GROQ_API_KEY", "").strip(), likert_scale=[1, 2])

class BoardConfig(BaseModel):
    chairman_model_id: str
    member_model_ids: List[str]


class QueryRequest(BaseModel):
    model_id: Optional[str] = None
    language: str = "en"
    depth: str
    subject: str
    chapter: str
    standard: str
    theme: str = ""
    qType: str
    num_questions: int = Field(default=1, ge=1, le=5)
    block: Optional[str] = None
    use_rag: bool = True
    use_citation: bool = False
    board: Optional[BoardConfig] = None
    enable_alignment: bool = True
    enable_dynamic_dropoff: bool = True
    enable_graph_expansion: bool = False
    enable_task_keywords: bool = True
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)


SUPPORTED_QTYPES = {
    "Multiple Choice (MCQ)",
    "Short Answer",
    "True/False",
}


def _validate_generation_request(req: QueryRequest) -> None:
    # Frontend already applies limits, but API requests can bypass UI checks.
    # Keep equivalent server-side validation to protect consistency and data quality.
    if req.qType not in SUPPORTED_QTYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported qType '{req.qType}'. Allowed values: "
                f"{', '.join(sorted(SUPPORTED_QTYPES))}"
            ),
        )


def _extract_option_markers(question_text: str) -> set:
    markers = set()
    for marker in ("A", "B", "C", "D"):
        pattern = rf"(?im)^\s*{marker}\s*[\).:\-]"
        if re.search(pattern, question_text or ""):
            markers.add(marker)
    return markers


def _check_question_type_alignment(requested_qtype: str, question_text: str, answer_text: str) -> Dict[str, Any]:
    text = (question_text or "").strip()
    answer = (answer_text or "").strip()

    is_tf = bool(re.search(r"(?i)\b(true|false)\b|सही\s*/\s*गलत|सही\s*या\s*गलत", text))
    option_markers = _extract_option_markers(text)
    is_mcq = len(option_markers) >= 4

    if requested_qtype == "Multiple Choice (MCQ)":
        if is_mcq:
            return {
                "type_match": True,
                "type_match_reason": "Detected MCQ option markers A/B/C/D in question.",
            }
        return {
            "type_match": False,
            "type_match_reason": "Expected MCQ options A/B/C/D, but they were not detected.",
        }

    if requested_qtype == "True/False":
        if is_tf:
            return {
                "type_match": True,
                "type_match_reason": "Detected True/False phrasing.",
            }
        return {
            "type_match": False,
            "type_match_reason": "Expected True/False phrasing, but it was not detected.",
        }

    if requested_qtype == "Short Answer":
        if is_mcq:
            return {
                "type_match": False,
                "type_match_reason": "Detected MCQ option markers; expected Short Answer format.",
            }
        if is_tf:
            return {
                "type_match": False,
                "type_match_reason": "Detected True/False phrasing; expected Short Answer format.",
            }
        return {
            "type_match": True,
            "type_match_reason": "No MCQ/True-False markers detected; treated as Short Answer.",
        }

    return {
        "type_match": False,
        "type_match_reason": "Question type validation was not available for this qType.",
    }


def detect_language(text: str) -> str:
    devanagari_pattern = re.compile(r"[\u0900-\u097F]")
    english_pattern = re.compile(r"[A-Za-z]")

    has_devanagari = bool(devanagari_pattern.search(text or ""))
    has_english = bool(english_pattern.search(text or ""))

    if has_english and has_devanagari:
        return "hien"
    if has_devanagari:
        return "hi"
    if has_english:
        return "en"
    return "unknown"


def _eval_vllm_a(req: QueryRequest, q: Dict[str, Any]) -> Dict[str, float]:
    ncert_score = param_ncert.evaluate(
        task_description=(
            "You are to determine whether the given question and answer pair "
            "is a standard NCERT 10th, 11th or 12th standard question or not."
        ),
        evaluation_parameter=(
            "You to rate how well it is aligned on a scale of 1 to 5. "
            "A score of 1 indicates low alignment while a score of 5 indicates high alignment."
        ),
        question=q["question"],
        answer="",
    )

    qtype_score = guardrails_qwen.evaluate(
        task_description=(f"You are to determine whether it is a/an {req.qType} question type or not."),
        evaluation_parameter=("You to rate whether the question satisfies all conditions on a scale of 1 to 2."),
        question=q["question"],
        answer=q["answer"],
    )

    guardrail_score = guardrails_qwen.evaluate(
        task_description=(
            "You are to determine whether the question and answer pair is appropriate, safe, and usable."
        ),
        evaluation_parameter=("You to rate whether the question is appropriate or not on a scale of 1 to 2."),
        question=q["question"],
        answer=q["answer"],
    )

    return {
        "ncert": round(float(ncert_score), 2),
        "qtype": round(float(qtype_score), 2),
        "guard": round(float(guardrail_score), 2),
    }


def _eval_vllm_b(req: QueryRequest, q: Dict[str, Any]) -> Dict[str, float]:
    validity_score = verification_llama.evaluate(
        task_description=(
            "You are to determine whether the given question and answer pair is valid or not. "
            "Try to solve the question without looking at the answer and then verify with the given answer."
        ),
        evaluation_parameter=("You to rate whether the question is appropriate or not on a scale of 1 to 2."),
        question=q["question"],
        answer=q["answer"],
    )

    bloom_score = llama_bloom.evaluate(
        task_description=(
            f"You are to evaluate the DoK level alignment of a question. The provided bloom level is {req.depth}."
        ),
        evaluation_parameter=("You to rate how well it is aligned on a scale of 1 to 5."),
        question=q["question"],
        answer=q["answer"],
    )

    return {
        "validity": round(float(validity_score), 2),
        "dok": round(float(bloom_score), 2),
    }


def _eval_language(req: QueryRequest, q: Dict[str, Any]) -> float:
    language_score = detect_language(q["question"] + "\n" + q["answer"])
    if language_score == req.language:
        return 2.0
    if language_score == "hien":
        return 1.5
    return 1.0


def get_alignment_score(req: QueryRequest, q: Dict[str, Any]) -> Dict[str, float]:
    scores_a = _eval_vllm_a(req, q)
    scores_b = _eval_vllm_b(req, q)
    language_score = _eval_language(req, q)

    result = {
        **scores_a,
        **scores_b,
        "language": language_score,
    }
    return result


class QuestionStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.lock = Lock()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS generated_questions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    subject TEXT,
                    chapter TEXT,
                    standard TEXT,
                    theme TEXT,
                    qtype TEXT,
                    depth TEXT,
                    language TEXT,
                    request_json TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    chunk_text TEXT,
                    chunk_source TEXT,
                    similarity REAL,
                    alignment_score REAL,
                    scores_json TEXT,
                    source_text_json TEXT,
                    source_meta_json TEXT,
                    board_metadata_json TEXT,
                    rubric_json TEXT,
                    type_match INTEGER,
                    type_match_reason TEXT,
                    is_rag INTEGER DEFAULT 0
                        , use_citation INTEGER DEFAULT 0
                )
                """
            )
            self._ensure_columns(conn)
            conn.commit()

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(generated_questions)")}
        required = {
            "alignment_score": "REAL",
            "scores_json": "TEXT",
            "source_text_json": "TEXT",
            "source_meta_json": "TEXT",
            "board_metadata_json": "TEXT",
            "rubric_json": "TEXT",
            "type_match": "INTEGER",
            "type_match_reason": "TEXT",
            "is_rag": "INTEGER DEFAULT 0",
            "use_citation": "INTEGER DEFAULT 0",
            "citation": "TEXT",
        }
        for column_name, column_type in required.items():
            if column_name not in existing:
                conn.execute(f"ALTER TABLE generated_questions ADD COLUMN {column_name} {column_type}")

    def save_batch(
        self,
        req: QueryRequest,
        questions: List[Dict[str, Any]],
        chunk_text: str,
        chunk_meta: Optional[Dict[str, Any]],
    ) -> List[str]:
        ids: List[str] = []
        now = datetime.now(timezone.utc).isoformat()
        chunk_source = (chunk_meta or {}).get("source_path")
        similarity = (chunk_meta or {}).get("similarity")

        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                for qa in questions:
                    row_id = str(uuid.uuid4())
                    ids.append(row_id)
                    source_text = qa.get("source_text")
                    source_meta = qa.get("source_meta")
                    board_metadata = qa.get("board_metadata")
                    rubric = qa.get("rubric")
                    scores = qa.get("scores")
                    type_match = qa.get("type_match")
                    type_match_reason = qa.get("type_match_reason")
                    conn.execute(
                        """
                        INSERT INTO generated_questions (
                            id, created_at, model_id, subject, chapter, standard, theme, qtype, depth, language,
                            request_json, question, answer, chunk_text, chunk_source, similarity,
                            alignment_score, scores_json, source_text_json, source_meta_json, board_metadata_json, rubric_json,
                            type_match, type_match_reason, is_rag, use_citation, citation
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row_id,
                            now,
                            req.model_id or "groq-llama-70b",
                            req.subject,
                            req.chapter,
                            req.standard,
                            req.theme,
                            req.qType,
                            req.depth,
                            req.language,
                            req.model_dump_json(),
                            qa.get("question", "").strip(),
                            qa.get("answer", "").strip(),
                            chunk_text,
                            chunk_source,
                            similarity,
                            qa.get("alignment_score"),
                            json.dumps(scores, ensure_ascii=False) if scores is not None else None,
                            json.dumps(source_text, ensure_ascii=False) if source_text is not None else None,
                            json.dumps(source_meta, ensure_ascii=False) if source_meta is not None else None,
                            json.dumps(board_metadata, ensure_ascii=False) if board_metadata is not None else None,
                            json.dumps(rubric, ensure_ascii=False) if rubric is not None else None,
                            (1 if type_match else 0) if type_match is not None else None,
                            type_match_reason,
                            1 if qa.get("is_rag") else 0,
                            1 if getattr(req, "use_citation", False) else 0,
                            qa.get("citation"),
                        ),
                    )
                conn.commit()

        return ids

    def list_questions(self, offset: int = 0, limit: int = 25) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                  SELECT id, created_at, model_id, subject, chapter, qtype, question, similarity, alignment_score,
                      type_match, type_match_reason, is_rag, use_citation, citation
                FROM generated_questions
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_question(self, row_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM generated_questions WHERE id = ?",
                (row_id,),
            ).fetchone()
        if not row:
            return None

        data = dict(row)
        for key in ("scores_json", "source_text_json", "source_meta_json", "board_metadata_json", "rubric_json"):
            if data.get(key):
                try:
                    data[key.replace("_json", "")] = json.loads(data[key])
                except Exception:
                    data[key.replace("_json", "")] = data[key]
        return data


def resolve_groq_model(model_id: Optional[str]) -> str:
    mapping = {
        "groq-llama-8b": "llama-3.1-8b-instant",
        "groq-llama-70b": "llama-3.3-70b-versatile",
        "rag-piped-groq-70b": "llama-3.3-70b-versatile",
        "groq-qwen-32b": "qwen/qwen3-32b",
        "groq-llama-guard": "meta-llama/llama-guard-4-12b",
        "groq-gpt-oss-120b": "openai/gpt-oss-120b",
        "groq-gpt-oss-20b": "openai/gpt-oss-20b",
    }
    model_id = _ensure_model_id(model_id, "groq-llama-70b")
    return mapping.get(model_id, "llama-3.3-70b-versatile")


# Add your local models to the global allowed set
ALLOWED_MODELS = {
    "groq-llama-8b", "groq-llama-70b", "rag-piped-groq-70b",
    "groq-qwen-32b", "groq-llama-guard", "groq-gpt-oss-120b", "groq-gpt-oss-20b",
    "ollama-gemma4-e4b", "ollama-olmo-3-7b", "ollama-phi4-mini", 
    "ollama-qwen-2b", "ollama-gemma4-e2b", "ollama-qwen-4b", "ollama-gemma4-31b"
}


app = FastAPI(title="BharatGen Minimal RAG API", version="0.1.0")

# --- UI MOUNTS AND ROUTES (Serving Separate HTML Files) ---
frontend_dir = (BASE_DIR / "../frontend").resolve()
frontend_static_dir = frontend_dir / "static"

if frontend_static_dir.exists():
    app.mount("/static", StaticFiles(directory=frontend_static_dir), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(frontend_dir / "index.html")

@app.get("/viewq")
async def serve_viewer():
    return FileResponse(frontend_dir / "viewq.html")
# ----------------------------------------------------------

try:
    from groq import Groq
except Exception as exc:  
    Groq = None
    _import_error = exc
else:
    _import_error = None


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
if Groq is not None and GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
    set_clients(groq_client=groq_client)
else:
    groq_client = None


retriever = MinimalRAGRetriever(RAG_STORE_DIR)
store = QuestionStore(DB_PATH)


def _build_source_meta(first_meta: Optional[Dict[str, Any]], retrieval_query: str) -> Dict[str, Any]:
    return {
        "query": retrieval_query,
        "k": DEFAULT_K,
        "pdf_path": (first_meta or {}).get("source_path"),
        "page": (first_meta or {}).get("page"),
        "title": (first_meta or {}).get("title"),
        "similarity": (first_meta or {}).get("similarity"),
        "word_count": (first_meta or {}).get("word_count"),
    }


async def _score_and_enrich_questions(
    req: QueryRequest,
    questions: List[Dict[str, Any]],
    retrieval_query: str,
    first_meta: Optional[Dict[str, Any]],
    board_metadata: Optional[Dict[str, Any]] = None,
    source_text: Optional[Dict[str, Any]] = None,
    is_rag: bool = False,
    generated_count: Optional[int] = None,
    requested_count: Optional[int] = None,
    generation_time_ms: Optional[float] = None,
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    source_meta = _build_source_meta(first_meta, retrieval_query)
    timestamp = datetime.now(timezone.utc).isoformat()

    async def _evaluate_single(question: Dict[str, Any]) -> Dict[str, Any]:
        type_check = _check_question_type_alignment(
            requested_qtype=req.qType,
            question_text=question.get("question", ""),
            answer_text=question.get("answer", ""),
        )
        
        # Alignment scoring - only if enabled
        scores = None
        alignment_score = None
        if req.enable_alignment:
            scores = await asyncio.to_thread(get_alignment_score, req, question)
            if (
                scores["guard"] < 1.5
                or scores["validity"] < 1.5
                or scores["qtype"] < 1.5
                or scores["language"] < 1.5
            ):
                alignment_score = 0.0
            else:
                alignment_score = round((scores["ncert"] + scores["dok"]) / 2, 2)

        item = dict(question)
        if scores is not None:
            item["scores"] = scores
        if alignment_score is not None:
            item["alignment_score"] = alignment_score
        item["is_rag"] = is_rag
        item["source_text"] = source_text or {"topic_chunk": "", "theme_chunk": ""}
        item["source_meta"] = source_meta
        item["generation_settings"] = {
            "use_rag": bool(req.use_rag),
            "use_citation": bool(getattr(req, "use_citation", False)),
            "enable_alignment": bool(req.enable_alignment),
            "enable_dynamic_dropoff": bool(req.enable_dynamic_dropoff),
            "enable_graph_expansion": bool(req.enable_graph_expansion),
            "enable_task_keywords": bool(req.enable_task_keywords),
            "temperature": float(req.temperature),
        }
        item["type_match"] = type_check["type_match"]
        item["type_match_reason"] = type_check["type_match_reason"]
        item["requested_count"] = requested_count if requested_count is not None else req.num_questions
        item["generated_count"] = generated_count if generated_count is not None else len(questions)
        item["count_warning"] = item["generated_count"] < item["requested_count"]
        item["timestamp"] = timestamp
        if generation_time_ms is not None:
            item["generation_time_ms"] = generation_time_ms
        if board_metadata is not None:
            item["board_metadata"] = board_metadata
        return item

    scored = await asyncio.gather(*[_evaluate_single(question) for question in questions])
    for item in scored:
        enriched.append(item)

    return enriched


@app.get("/health")
def health() -> Dict[str, Any]:
    llm_provider = os.getenv("LLM_PROVIDER", "groq").lower()
    is_ok = groq_client is not None if llm_provider == "groq" else True 

    return {
        "ok": is_ok,
        "provider": llm_provider,
        "groq_ready": groq_client is not None,
        "rag_store": str(RAG_STORE_DIR),
        "documents_loaded": len(retriever.records),
        "default_k": DEFAULT_K,
        "chunk_word_limits": [MIN_CHUNK_WORDS + 1, MAX_CHUNK_WORDS - 1],
        "import_error": str(_import_error) if _import_error else None,
    }


@app.post("/ask")
async def ask(req: QueryRequest) -> List[Dict[str, Any]]:
    start_time = time.time()  # Start timer
    
    if os.getenv("LLM_PROVIDER", "groq").lower() == "groq" and groq_client is None:
        raise HTTPException(status_code=500, detail="Groq client not available. Set GROQ_API_KEY and install groq.")

    _validate_generation_request(req)

    model_id = _normalize_legacy_generation_model_id(_ensure_model_id(req.model_id, "groq-llama-70b"))
    if req.board is None and model_id not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model_id '{model_id}'. Use a Groq model only.")

    if req.board:
        chairman_model = _normalize_legacy_generation_model_id(_ensure_model_id(req.board.chairman_model_id, "groq-llama-70b"))
        member_models = [_normalize_legacy_generation_model_id(_ensure_model_id(model, "groq-llama-8b")) for model in req.board.member_model_ids]
        if chairman_model not in ALLOWED_MODELS:
            raise HTTPException(status_code=400, detail=f"Unsupported chairman model '{chairman_model}'. Use a Groq model only.")
        invalid_members = [model for model in member_models if model not in ALLOWED_MODELS]
        if invalid_members:
            raise HTTPException(status_code=400, detail=f"Unsupported member models: {', '.join(invalid_members)}")

        if chairman_model in member_models:
            raise HTTPException(status_code=400, detail="Chairman model cannot be in member list")
        if not member_models:
            raise HTTPException(status_code=400, detail="At least one board member is required")

        council_result = await run_council_flow(
            chairman_model_id=chairman_model,
            member_model_ids=member_models,
            language=req.language,
            subject=req.subject,
            chapter=req.chapter,
            theme=req.theme,
            qType=req.qType,
            depth=req.depth,
            num_questions=get_generation_question_count(req.depth, req.num_questions),
            use_rag=req.use_rag,
            use_citation=req.use_citation,
            enable_dynamic_dropoff=req.enable_dynamic_dropoff,
            enable_graph_expansion=req.enable_graph_expansion,
            temperature=req.temperature,
        )

        questions = parse_ai_output(council_result.get("final_output", ""))
        if not questions:
            chairman_proposal = council_result.get("chairman_proposal") or ""
            if chairman_proposal:
                questions = parse_ai_output(chairman_proposal)

        if not questions:
            raise HTTPException(status_code=500, detail="Council flow completed but no parsable questions were produced.")

        requested_count = get_generation_question_count(req.depth, req.num_questions)
        questions = questions[: requested_count]
        generated_count = len(questions)
        source_chunks = council_result.get("source_chunks") or {"topic_chunk": "", "theme_chunk": ""}
        source_meta = council_result.get("source_meta") or {}
        board_metadata = {
            "chairman": chairman_model,
            "members": member_models,
            "language": req.language,
            "chairman_proposal": council_result.get("chairman_proposal"),
            "member_opinions": council_result.get("member_opinions", []),
        }

        enriched = await _score_and_enrich_questions(
            req=req,
            questions=questions,
            retrieval_query=" ".join([req.chapter or "", req.theme or ""]).strip() or req.subject,
            first_meta=source_meta,
            board_metadata=board_metadata,
            source_text=source_chunks,
            is_rag=bool(req.use_rag or req.use_citation),
            generated_count=generated_count,
            requested_count=get_generation_question_count(req.depth, req.num_questions),
            generation_time_ms=round((time.time() - start_time) * 1000, 2),
        )

        saved_ids = store.save_batch(req, enriched, chunk_text=(source_chunks.get("topic_chunk") or "") + ("\n\n" + source_chunks.get("theme_chunk") if source_chunks.get("theme_chunk") else ""), chunk_meta=source_meta)
        for i, row_id in enumerate(saved_ids):
            if i < len(enriched):
                enriched[i]["id"] = row_id
        return enriched

    retrieval_query = " ".join([req.chapter or "", req.theme or ""]).strip() or req.subject
    chunk_text = ""
    metas: List[Dict[str, Any]] = []

    # Citation mode takes precedence over generic RAG retrieval when enabled
    if req.use_citation:
        # Citation-based retrieval: pick a random citation verbatim from the selected block
        chunk_text, metas = retriever.retrieve_citation(
            query=retrieval_query,
            subject=req.subject,
            chapter=req.chapter,
            standard=req.standard,
            block=req.block,
            language=req.language,
        )
        if not chunk_text:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No citation entries found in the selected Block. Ensure the Block contains a citations.json file with usable quotes."
                ),
            )
    elif req.use_rag:
        # Standard semantic RAG retrieval
        chunk_text, metas = retriever.retrieve(
            query=retrieval_query,
            subject=req.subject,
            chapter=req.chapter,
            standard=req.standard,
            block=req.block,
            k=DEFAULT_K,
            enable_dynamic_dropoff=req.enable_dynamic_dropoff,
            enable_graph_expansion=req.enable_graph_expansion,
        )
        if not chunk_text:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No valid chunks found after word-count filtering (must be >5 and <5000 words). "
                    "Try different chapter/theme/subject or disable RAG."
                ),
            )

    prompt = build_prompt_from_request(req, chunk_text)
    raw_output = await run_model(model_id, prompt, req=req)
    questions = parse_ai_output(raw_output)

    if not questions:
        questions = [{"question": raw_output.strip(), "answer": ""}]

    requested_count = get_generation_question_count(req.depth, req.num_questions)
    questions = questions[: requested_count]
    generated_count = len(questions)
    first_meta = metas[0] if metas else None
    source_text = {"topic_chunk": chunk_text, "theme_chunk": ""} if (req.use_rag or req.use_citation) else {"topic_chunk": "", "theme_chunk": ""}

    enriched = await _score_and_enrich_questions(
        req=req,
        questions=questions,
        retrieval_query=retrieval_query,
        first_meta=first_meta,
        source_text=source_text,
        is_rag=bool(req.use_rag or req.use_citation),
        generated_count=generated_count,
        requested_count=requested_count,
        generation_time_ms=round((time.time() - start_time) * 1000, 2),
    )

    saved_ids = store.save_batch(req, enriched, chunk_text=chunk_text if (req.use_rag or req.use_citation) else "", chunk_meta=first_meta)
    for i, row_id in enumerate(saved_ids):
        if i < len(enriched):
            enriched[i]["id"] = row_id

    return enriched


@app.get("/api/questions")
def list_saved_questions(offset: int = 0, limit: int = 25) -> List[Dict[str, Any]]:
    rows = store.list_questions(offset=offset, limit=limit)
    for row in rows:
        if row.get("type_match") is not None:
            row["type_match"] = bool(row["type_match"])
    return rows


@app.get("/api/question/{qid}")
def get_saved_question(qid: str) -> Dict[str, Any]:
    row = store.get_question(qid)
    if not row:
        raise HTTPException(status_code=404, detail="Question not found")
    
    if row.get("type_match") is not None:
        row["type_match"] = bool(row["type_match"])

    try:
        req_json = json.loads(row.get("request_json") or "{}") if row.get("request_json") else {}
    except Exception:
        req_json = {}

    # UI compatibility: normalize request metadata for dashboards.
    row["req"] = {
        "model": row.get("model_id"),
        "subject": row.get("subject"),
        "chapter": row.get("chapter"),
        "qType": row.get("qtype"),
        "num_questions": req_json.get("num_questions"),
        "enable_task_keywords": req_json.get("enable_task_keywords", True),
    }
    row["scores"] = row.get("scores") or {"similarity": row.get("similarity"), "alignment_score": row.get("alignment_score")}
    return row

def _get_books_root():
    """Books root for PDF serving; must match ingest BOOKS_ROOT."""
    env_path = os.getenv("BHARATGEN_BOOKS_PATH")
    if env_path:
        return Path(env_path).resolve()
    project_root = BASE_DIR.parent
    books_dir = project_root / "books"
    if books_dir.is_dir():
        return books_dir.resolve()
    return (project_root / "data").resolve()


def _extract_block_label(file_name: str) -> str:
    stem = Path(file_name).stem
    match = re.search(r"block\s*(\d+)", stem, re.IGNORECASE)
    if match:
        return f"Block {int(match.group(1))}"
    return stem

@app.get("/course-blocks")
async def list_course_blocks():
    """Dynamically build the UI dropdowns directly from the indexed RAG folders."""
    if not RAG_STORE_DIR.exists() or not RAG_STORE_DIR.is_dir():
        return {"courses": []}

    course_rows = []
    
    for course_dir in sorted([p for p in RAG_STORE_DIR.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        blocks = []
        
        search_dirs = [course_dir]
        if (course_dir / "egyankosh").is_dir():
            search_dirs.append(course_dir / "egyankosh")
            
        for search_dir in search_dirs:
            for block_dir in search_dir.iterdir():
                if block_dir.is_dir() and "block" in block_dir.name.lower():
                    # Do not truncate the name using regex - keep the full folder name.
                    blocks.append(block_dir.name)
        
        # Sort blocks intelligently (Block 1, Block 2, etc.)
        uniq_blocks = sorted(
            set(blocks),
            key=lambda b: (int(re.search(r"\d+", b).group()) if re.search(r"\d+", b) else 10_000, b.lower())
        )

        if not uniq_blocks:
            continue

        course_rows.append(
            {
                "course_name": course_dir.name,
                "blocks": uniq_blocks,
            }
        )

    return {"courses": course_rows}