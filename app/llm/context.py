"""Backward-compatible import for the canonical prompt context type.

Prompt assembly lives under :mod:`app.llm.prompts`; this alias preserves the
original import path for integrations that imported ``PromptContext`` directly.
"""

from __future__ import annotations

from app.llm.prompts.context import PromptContext

__all__ = ["PromptContext"]
