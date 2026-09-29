"""Small shared helpers used across services and routes."""

import uuid
from datetime import datetime, timezone


def new_id() -> str:
    """Generate a UUID4 string — used for document ids and Qdrant point ids."""
    return str(uuid.uuid4())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def success_response(data=None):
    return {"success": True, "data": data if data is not None else {}}


def error_response(code: str, message: str):
    return {"success": False, "error": {"code": code, "message": message}}
