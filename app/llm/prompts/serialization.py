"""Safe, deterministic serialization for untrusted values embedded in prompts."""

from __future__ import annotations

import json
from typing import Any


def serialize_untrusted(value: Any) -> str:
    """Encode prompt data as JSON without allowing delimiter-shaped text through.

    JSON already prevents quotes and newlines from changing the surrounding shape.
    Escaping the HTML-significant characters as JSON unicode escapes also prevents a
    value such as ``</memory_data>`` from visually terminating a prompt data block.
    The line-separator escapes keep provider-side parsing consistent as well.
    """
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
