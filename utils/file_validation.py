"""File upload validation: extension, size, emptiness, and basic sanity checks."""

import os
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from config.settings import settings


class ValidationError(Exception):
    """Raised when an uploaded file fails validation."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass
class ValidatedFile:
    safe_filename: str
    extension: str
    size_bytes: int


def sanitize_filename(filename: str) -> str:
    """Strip path components and dangerous characters to prevent path traversal."""
    # Keep only the final path segment, defeating any directory traversal attempt.
    name = PurePosixPath(filename.replace("\\", "/")).name
    name = name.strip()
    if not name or name in (".", ".."):
        raise ValidationError("INVALID_FILENAME", "The uploaded file has no valid filename.")
    # Allow letters, numbers, spaces, dots, hyphens, underscores only.
    name = re.sub(r"[^A-Za-z0-9 ._-]", "_", name)
    # Collapse repeated dots (defends against sneaky "..").
    name = re.sub(r"\.{2,}", ".", name)
    return name[:255]


def validate_upload(file_storage) -> ValidatedFile:
    """
    Validate a Werkzeug FileStorage object.

    Checks extension, declared MIME type (best-effort — browsers are not
    always accurate), and file size. Raises ValidationError with a
    user-friendly message on any failure.
    """
    if file_storage is None or file_storage.filename == "":
        raise ValidationError("NO_FILE", "No file was provided.")

    safe_name = sanitize_filename(file_storage.filename)
    ext = os.path.splitext(safe_name)[1].lower()

    if ext not in settings.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(settings.ALLOWED_EXTENSIONS))
        raise ValidationError(
            "UNSUPPORTED_TYPE",
            f"'{ext or 'unknown'}' files are not supported. Allowed types: {allowed}.",
        )

    declared_mime = (file_storage.mimetype or "").lower()
    allowed_mimes = settings.ALLOWED_MIME_TYPES.get(ext, set())
    if declared_mime and allowed_mimes and declared_mime not in allowed_mimes:
        # Best-effort check only — some browsers/OSes send generic MIME types,
        # so we warn via rejection only for clearly wrong types.
        if not declared_mime.startswith("application/") and not declared_mime.startswith("text/"):
            raise ValidationError(
                "MIME_MISMATCH",
                f"The file's content type ('{declared_mime}') does not match a {ext} file.",
            )

    # Determine size without loading the whole file into memory.
    file_storage.stream.seek(0, os.SEEK_END)
    size_bytes = file_storage.stream.tell()
    file_storage.stream.seek(0)

    if size_bytes == 0:
        raise ValidationError("EMPTY_FILE", "The uploaded file is empty.")

    if size_bytes > settings.MAX_UPLOAD_SIZE_BYTES:
        raise ValidationError(
            "FILE_TOO_LARGE",
            f"File exceeds the {settings.MAX_UPLOAD_SIZE_MB}MB upload limit.",
        )

    return ValidatedFile(safe_filename=safe_name, extension=ext, size_bytes=size_bytes)
