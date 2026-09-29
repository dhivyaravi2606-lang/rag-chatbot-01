"""
Flask application entrypoint.

Endpoints:
  GET  /                     -> chat/dashboard UI
  GET  /api/health           -> service health check (Qdrant + Gemini)
  POST /api/documents        -> upload + ingest a document
  GET  /api/documents        -> list documents + knowledge base stats
  DELETE /api/documents/<id> -> delete a document and its vectors
  POST /api/chat             -> ask a question, get a grounded answer
"""

import logging
import uuid

from flask import Flask, jsonify, render_template, request

from config.settings import settings
from services import document_service, gemini_service, qdrant_service, rag_service
from services.document_service import ExtractionError
from services.embedding_service import EmbeddingError
from services.gemini_service import GenerationError
from services.qdrant_service import QdrantServiceError
from services.rag_service import IngestionError
from utils.file_validation import ValidationError, validate_upload
from utils.helpers import error_response, new_id, success_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("rag_app")


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = settings.FLASK_SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = settings.MAX_UPLOAD_SIZE_BYTES

    settings.UPLOAD_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

    problems = settings.validate()
    if problems:
        for p in problems:
            logger.warning("Configuration warning: %s", p)

    # ---------------------------------------------------------------- UI

    @app.route("/")
    def index():
        return render_template("index.html")

    # ------------------------------------------------------------ health

    @app.route("/api/health")
    def health():
        qdrant_ok = qdrant_service.health_check()
        gemini_ok = gemini_service.health_check()
        stats = document_service.knowledge_base_stats()
        return jsonify(
            success_response(
                {
                    "qdrant": {"connected": qdrant_ok},
                    "gemini": {"configured": gemini_ok},
                    "knowledge_base": stats,
                }
            )
        )

    # -------------------------------------------------------- documents

    @app.route("/api/documents", methods=["GET"])
    def list_documents():
        docs = document_service.list_documents()
        return jsonify(success_response({"documents": docs}))

    @app.route("/api/documents", methods=["POST"])
    def upload_document():
        file_storage = request.files.get("file")
        try:
            validated = validate_upload(file_storage)
        except ValidationError as exc:
            return jsonify(error_response(exc.code, exc.message)), 400

        doc_id = new_id()
        stored_name = f"{doc_id}{validated.extension}"
        stored_path = settings.UPLOAD_STORAGE_PATH / stored_name

        try:
            file_storage.save(stored_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to save uploaded file")
            return jsonify(error_response("SAVE_FAILED", "Could not save the uploaded file.")), 500

        document_service.create_document_record(
            doc_id=doc_id,
            filename=validated.safe_filename,
            file_type=validated.extension.lstrip("."),
            file_size=validated.size_bytes,
        )

        try:
            chunk_count = rag_service.ingest_document(
                doc_id=doc_id,
                file_path=stored_path,
                extension=validated.extension,
                original_filename=validated.safe_filename,
            )
        except (ExtractionError, EmbeddingError, QdrantServiceError, IngestionError) as exc:
            return (
                jsonify(error_response(exc.code, exc.message)),
                422,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected error during ingestion")
            return jsonify(error_response("INGESTION_FAILED", "Unexpected error while processing the document.")), 500

        doc = document_service.get_document(doc_id)
        return jsonify(success_response({"document": doc, "chunks_indexed": chunk_count})), 201

    @app.route("/api/documents/<doc_id>", methods=["DELETE"])
    def delete_document(doc_id):
        doc = document_service.get_document(doc_id)
        if not doc:
            return jsonify(error_response("NOT_FOUND", "Document not found.")), 404

        try:
            qdrant_service.delete_document_vectors(doc_id)
        except QdrantServiceError as exc:
            return jsonify(error_response(exc.code, exc.message)), 502

        # Remove the stored file, if present, then the metadata record.
        for ext in settings.ALLOWED_EXTENSIONS:
            candidate = settings.UPLOAD_STORAGE_PATH / f"{doc_id}{ext}"
            if candidate.exists():
                try:
                    candidate.unlink()
                except OSError:
                    logger.warning("Could not remove stored file for %s", doc_id)

        document_service.delete_document_record(doc_id)
        return jsonify(success_response({"deleted": doc_id}))

    # -------------------------------------------------------------- chat

    @app.route("/api/chat", methods=["POST"])
    def chat():
        body = request.get_json(silent=True) or {}
        question = (body.get("question") or "").strip()
        history_in = body.get("history") or []

        if not question:
            return jsonify(error_response("EMPTY_QUESTION", "Please enter a question.")), 400
        if len(question) > 2000:
            return jsonify(error_response("QUESTION_TOO_LONG", "Question is too long (max 2000 characters).")), 400

        # history_in: [{"role": "user"|"assistant", "text": "..."}]
        history = [(h.get("role"), h.get("text", "")) for h in history_in if h.get("text")]

        try:
            result = rag_service.answer_question(question, history)
        except (EmbeddingError, QdrantServiceError, GenerationError, IngestionError) as exc:
            return jsonify(error_response(exc.code, exc.message)), 502
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected error answering question")
            return jsonify(error_response("CHAT_FAILED", "Unexpected error while answering your question.")), 500

        return jsonify(success_response(result))

    # ------------------------------------------------------ error pages

    @app.errorhandler(413)
    def too_large(_exc):
        return (
            jsonify(error_response("FILE_TOO_LARGE", f"File exceeds the {settings.MAX_UPLOAD_SIZE_MB}MB upload limit.")),
            413,
        )

    @app.errorhandler(404)
    def not_found(_exc):
        return jsonify(error_response("NOT_FOUND", "Resource not found.")), 404

    @app.errorhandler(500)
    def server_error(_exc):
        logger.exception("Unhandled server error")
        return jsonify(error_response("INTERNAL_ERROR", "An internal server error occurred.")), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=settings.PORT, debug=settings.FLASK_DEBUG)
