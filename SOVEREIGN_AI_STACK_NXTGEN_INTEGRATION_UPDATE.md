# Sovereign AI Stack Showcase – Technical Update for NxtGen Integration

**Prepared for:** Smrutikant Nayak, NxtGen Team  
**From:** Venkat / IIT Bombay team  
**Date:** 2 February 2026  
**Re:** Integration questions – API, parameters, conversation limits, and repository

**Current configuration:** Groq-only models; theme selector removed from UI (default `"general"`).

---

## 1. API Integration: Same API or Bridge/Middleware?

**Answer:** There is a **bridge/middleware API**, not the raw AMITH API.

The BharatGen application exposes its own REST API that:
- Accepts parameters such as **subject**, **chapter**, **depth**, **qType**, **language**, and board configuration
- Uses **Groq models only** (Llama, GPT OSS variants) – no local/Param models
- Optionally accepts **theme** (default: `"general"`) for RAG retrieval – not shown in UI
- Returns structured JSON responses suitable for the chat/explore UI

NxtGen’s chat page should integrate with **this BharatGen backend API**, not the underlying LLM provider APIs directly.

---

## 2. Main APIs

### A. Question Generation: `POST /ask`

Generates NCERT-aligned questions with subject, chapter, etc. Uses Groq models only.

**Request Body (JSON):**

```json
{
  "board": {
    "chairman_model_id": "groq-llama-70b",
    "member_model_ids": ["groq-llama-8b"]
  },
  "language": "en",
  "depth": "DOK level 1: Recall & Reproduction",
  "subject": "Physics",
  "chapter": "Motion in a Straight Line",
  "qType": "Multiple Choice (MCQ)",
  "num_questions": 2
}
```

**Note:** `theme` is optional (default `"general"`). Used for RAG retrieval; omit or set to `"general"` unless a specific theme (e.g. cricket, IPL) is needed.

**cURL example:**

```bash
curl -X POST "https://<BASE_URL>/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "board": {
      "chairman_model_id": "groq-llama-70b",
      "member_model_ids": ["groq-llama-8b"]
    },
    "language": "en",
    "depth": "DOK level 1: Recall & Reproduction",
    "subject": "Physics",
    "chapter": "Motion in a Straight Line",
    "qType": "Multiple Choice (MCQ)",
    "num_questions": 2
  }'
```

**Response:** Array of objects with `question`, `answer`, `source_text`, `source_meta`, `board_metadata`, etc.

---

### B. Chat with Source Context: `POST /explore/chat`

For the **chat page** – answers grounded in a given source chunk. Supports multi-turn conversation.

**Request Body (JSON):**

```json
{
  "chunk_text": "The source material text (topic chunk + theme chunk from RAG retrieval, if any)...",
  "pdf_path": "English/Physics/Class-11/English_Physics_Class-11.pdf",
  "page": 5,
  "messages": [
    { "role": "user", "content": "What is the formula for displacement?" },
    { "role": "assistant", "content": "Displacement s = ut + (1/2)at²..." },
    { "role": "user", "content": "Can you give an example using cricket?" }
  ]
}
```

**cURL example:**

```bash
curl -X POST "https://<BASE_URL>/explore/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "chunk_text": "The NCERT source material text for the topic and theme...",
    "pdf_path": "English/Physics/Class-11/English_Physics_Class-11.pdf",
    "page": 5,
    "messages": [
      {"role": "user", "content": "What is displacement?"},
      {"role": "assistant", "content": "Displacement is the change in position..."},
      {"role": "user", "content": "Give a cricket example."}
    ]
  }'
```

**Response:**

```json
{
  "reply": "The assistant's response grounded in the source material."
}
```

---

## 3. Follow-up Questions in the Same Conversation

**Yes.** Users can send follow-up questions in the same conversation.

The `/explore/chat` API accepts a `messages` array. The client should:

1. Maintain conversation history in the UI
2. Append each new user message and assistant reply to `messages`
3. Send the full `messages` array with every request

The backend concatenates all messages into the prompt and instructs the model to answer only from the source material.

---

## 4. How Many Messages Per Conversation?

There is **no hard-coded limit** in the backend. The entire `messages` array is included in the prompt.

Practical limits come from:
- Model context windows (e.g., 8K–32K tokens)
- Chunk size of `chunk_text` (often ~800–1600 chars per chunk)

**Recommendation:** Keep the last **10–20** message pairs (user + assistant) per conversation. Beyond that, consider:
- Summarizing older messages, or
- Trimming to the last N messages to avoid hitting token limits

---

## 5. Parameters Overview

| API | Parameters |
|-----|------------|
| **POST /ask** | `board` (chairman_model_id, member_model_ids), `language`, `depth`, `subject`, `chapter`, `qType`, `num_questions`; `theme` optional (default `"general"`) |
| **POST /explore/chat** | `chunk_text`, `pdf_path`, `page`, `messages` |

**Available Groq models:** `groq-llama-8b`, `groq-llama-70b`, `rag-piped-groq-70b`, `groq-llama-guard`, `groq-gpt-oss-120b`, `groq-gpt-oss-20b`

The `/ask` API also supports single-model mode with `model_id` instead of `board` for backward compatibility.

---

## 6. Repository and Deployment

The current codebase is the BharatGen Question Generator (also referenced as “bharatgen-ibm-yojaka-llmquestion-board” in docs).

**Recommendation for NxtGen:**

1. **Same repo:** Use the existing BharatGen repo if that is the agreed integration source.
2. **New repo:** If a separate repo is created for NxtGen deployment, ensure it includes:
   - Latest `backend/main.py`, `model_runner.py`, `council.py`, `ncert_rag_pipe/`
   - Latest `frontend/index.html`, `frontend/explore.html`
   - `books/` (or equivalent) for NCERT PDFs
   - `indexes/` for RAG vector DB (after running ingest)

The final repo URL and deployment target for NxtGen infra should be confirmed between IIT Bombay and NxtGen teams.

---

## 7. Additional Endpoints

- **GET /** – Serves the main question generator UI
- **GET /explore.html** – Serves the Explore/Chat UI
- **GET /chapters?subject=&language=** – Returns chapter names for a subject (en/hi)
- **GET /api/pdf?path=** – Serves PDF files from the books root
- **GET /health** – Health check; returns `{"ok": true, "message": "Groq-only mode"}` when Groq client is initialized

---

## 8. Environment Configuration

The backend requires:
- **`GROQ_API_KEY`** (required) – for all Groq models (question generation and explore/chat)

Optional:
- `BHARATGEN_BOOKS_PATH` – path to NCERT PDFs (default: project `books/` or `data/`)
- `EXPLORE_CHAT_MODEL` – model for explore/chat (default: `groq-llama-8b`)

---

*For integration support or a call to walk through the APIs, please reach out to the IIT Bombay / BharatGen team.*
