# Athenaeum — RAG Knowledge Assistant

A production-ready Retrieval-Augmented Generation (RAG) application. Upload PDF, DOCX, or TXT documents, and ask questions that are answered strictly from what you've uploaded — with sources shown for every grounded answer.

Backend: Python + Flask. LLM + embeddings: Google Gemini API. Vector store: Qdrant. Frontend: vanilla HTML5/CSS3/JS with a dark glassmorphism UI (no frontend framework).

---

## 1. Architecture

**Ingestion pipeline** (`services/rag_service.py::ingest_document`)

```
Upload → Validate (ext/MIME/size) → Extract text (pypdf / python-docx / utf-8)
       → Clean → Chunk (configurable size/overlap) → Embed (Gemini, RETRIEVAL_DOCUMENT)
       → Upsert to Qdrant (with document_id, filename, page, chunk metadata)
```

**Question-answering pipeline** (`services/rag_service.py::answer_question`)

```
Question → Validate → Embed (Gemini, RETRIEVAL_QUERY) → Qdrant similarity search
         → Filter by SIMILARITY_THRESHOLD → Build context block → Gemini generate_content
         → Grounded answer + source list (filename, page, relevance %)
```

Ingestion and Q&A are implemented as independent functions with no shared mutable state, per the separation-of-concerns requirement.

### Project layout

```
rag-app/
├── app.py                     Flask app + REST endpoints
├── requirements.txt
├── .env.example
├── config/
│   ├── settings.py            All configuration, env-var driven
│   └── rag_prompt.txt         Editable RAG system prompt template
├── services/
│   ├── document_service.py    Text extraction (PDF/DOCX/TXT) + SQLite metadata store
│   ├── embedding_service.py   Gemini embeddings (documents + queries)
│   ├── gemini_service.py      Gemini generation (grounded answers)
│   ├── qdrant_service.py      Collection lifecycle, upsert, search, delete
│   └── rag_service.py         Orchestrates ingestion + Q&A pipelines
├── utils/
│   ├── chunking.py            Configurable, boundary-aware text chunking
│   ├── file_validation.py     Extension/MIME/size validation, filename sanitizing
│   └── helpers.py             IDs, timestamps, response envelopes
├── templates/index.html
└── static/{style.css, app.js}
```

Document metadata (filename, size, status, chunk count) lives in a small local SQLite file (`data/documents.db`) — Qdrant only stores what it's meant to: vectors + chunk payloads. This keeps document listing/deletion fast and avoids overloading the vector store with bookkeeping data.

---

## 2. Features

- Drag-and-drop or click-to-browse upload, with a live status flow: uploading → extracting → chunking → embedding → indexing → completed (or failed, with a reason).
- PDF (including per-page text + page-number citations), DOCX (paragraphs + tables), and TXT (encoding-safe) extraction.
- Configurable chunking (size, overlap), retrieval top-k, and similarity threshold — no arbitrary hard-coded splitting.
- Every chunk stores `document_id`, `filename`, `chunk_id`, `chunk_index`, `page_number` (when available).
- Chat UI: markdown rendering, code blocks, timestamps, auto-scroll, Enter-to-send / Shift+Enter for newline, loading indicator, inline error bubbles, clear-conversation.
- Source cards under every grounded answer: document name, page (if available), relevance %. Never fabricated — built directly from Qdrant's returned matches.
- Document management panel: type, size, upload date, live status, chunk count, delete (which also purges that document's vectors from Qdrant — no orphaned vectors).
- System status panel with **real** health checks (Qdrant connectivity + collection reachability, Gemini API key configured) — never a hard-coded "Connected".
- Consistent JSON envelopes (`{success, data}` / `{success:false, error:{code,message}}`), no stack traces returned to the client.
- Dark glassmorphism UI, fully responsive: sidebar/chat collapse into a tab switcher under 900px, no horizontal scroll, touch-sized controls.

---

## 3. Requirements

- Python 3.10+
- A Gemini API key ([Google AI Studio](https://aistudio.google.com/apikey))
- A Qdrant instance — either a free [Qdrant Cloud](https://cloud.qdrant.io) cluster, or a local instance:
  ```bash
  docker run -p 6333:6333 qdrant/qdrant
  ```

---

## 4. Environment variables

Copy `.env.example` to `.env` and fill in real values. Never commit `.env`.

| Variable | Purpose | Default |
|---|---|---|
| `GEMINI_API_KEY` | Gemini API key | *(required)* |
| `GEMINI_MODEL` | Generation model | `gemini-2.5-flash` |
| `EMBEDDING_MODEL` | Embedding model | `gemini-embedding-001` |
| `EMBEDDING_DIMENSIONS` | Output vector size (must match Qdrant collection) | `768` |
| `QDRANT_URL` | Qdrant endpoint | `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant Cloud API key (blank for local) | *(empty)* |
| `QDRANT_COLLECTION_NAME` | Collection name | `rag_knowledge_base` |
| `CHUNK_SIZE` | Target characters per chunk | `1200` |
| `CHUNK_OVERLAP` | Character overlap between chunks | `200` |
| `TOP_K` | Chunks retrieved per question | `5` |
| `SIMILARITY_THRESHOLD` | Minimum cosine score to use a chunk | `0.55` |
| `MAX_UPLOAD_SIZE_MB` | Upload size limit | `20` |
| `FLASK_SECRET_KEY` | Flask session secret | *(change me)* |
| `PORT` | Server port | `5000` |

`EMBEDDING_DIMENSIONS` is applied when the app creates the Qdrant collection *and* when it requests embeddings from Gemini, so the two always match. If you change it after the collection already exists, delete/recreate the collection (or point at a new `QDRANT_COLLECTION_NAME`) — Qdrant collections are fixed-dimension.

---

## 5. Installation & running locally

```bash
cd rag-app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env: set GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY (if using Qdrant Cloud)

python app.py
```

Open `http://localhost:5000`. The document store (SQLite) and the Qdrant collection are created automatically on first use — nothing to run manually beyond starting Qdrant itself.

---

## 6. API overview

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Qdrant/Gemini status + knowledge-base stats |
| `GET` | `/api/documents` | List documents |
| `POST` | `/api/documents` | Upload + ingest a file (`multipart/form-data`, field `file`) |
| `DELETE` | `/api/documents/<id>` | Delete a document and its vectors |
| `POST` | `/api/chat` | `{ "question": "...", "history": [{"role","text"}, ...] }` → grounded answer + sources |

All responses: `{"success": true, "data": {...}}` or `{"success": false, "error": {"code": "...", "message": "..."}}`.

---

## 7. RAG quality testing

Suggested manual test sequence once a document is uploaded:

- **A — Directly answered question**: ask something explicitly stated in the document. Expect a grounded answer with a high-relevance source card.
- **B — Unrelated question**: ask something with no connection to the uploaded content. Expect the assistant to say the knowledge base doesn't contain enough information, with no (or very low-relevance) sources.
- **C — Partially related question**: ask something adjacent to the content. Expect a partial answer that names its limits.
- **D — Multi-document question**: upload two documents covering related ground and ask a question spanning both. Expect source cards from more than one file.

Tune `SIMILARITY_THRESHOLD` if you see too many irrelevant chunks (raise it) or too many "not enough information" responses on genuinely relevant questions (lower it slightly).

---

## 8. Try it with the included sample document

`sample_documents/emu_war_1932.txt` is a self-contained article about the Great Emu War of 1932 — a real, specific, slightly obscure historical event, chosen so it's easy to tell whether an answer actually came from the retrieved text (rather than the model's general knowledge). Upload it, then try:

- "Why did the military get involved with the emus?"
- "Why were the emus so hard to defeat?"
- "How many emus were confirmed killed and how much ammunition did that take?"
- "What happened to Major Meredith's career afterward?" *(not in the document — should trigger a "not enough information" response)*

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Vector store shows "Unavailable" | Qdrant not running / wrong `QDRANT_URL` | Start Qdrant, check URL/port/API key |
| Language model shows "Not configured" | `GEMINI_API_KEY` missing/blank | Set it in `.env` and restart |
| Upload fails with `UNSUPPORTED_TYPE` | File isn't `.pdf`/`.docx`/`.txt` | Convert or use a supported format |
| Upload fails with `NO_TEXT_FOUND` | Scanned/image-only PDF | Needs OCR first; not handled by this app |
| Chat always says "not enough information" | Threshold too high, or nothing relevant was uploaded | Lower `SIMILARITY_THRESHOLD`, confirm the document actually covers the topic |
| `413` on upload | File exceeds `MAX_UPLOAD_SIZE_MB` | Raise the limit or use a smaller file |
| Vector dimension errors from Qdrant | `EMBEDDING_DIMENSIONS` changed after the collection was created | Use a new `QDRANT_COLLECTION_NAME` or delete the old collection |

---

## 10. Deployment considerations

- Run behind a production WSGI server (e.g. `gunicorn app:app`), not Flask's dev server.
- Put a reverse proxy (nginx/Caddy) in front for TLS termination.
- Set `FLASK_DEBUG=false` (default) in production — debug mode must never be on publicly.
- `data/` (SQLite file + uploaded originals) should sit on persistent storage if you deploy on ephemeral compute (containers) — otherwise document metadata and cached originals are lost on restart, even though vectors remain safely in Qdrant.
- Configure CORS explicitly (e.g. `flask-cors` with an allow-list) only if the frontend will be served from a different origin than the API; this app serves both from the same Flask process, so no CORS setup is needed by default.
- Rotate `FLASK_SECRET_KEY` and API keys via your platform's secret manager, not the `.env` file, in real deployments.
