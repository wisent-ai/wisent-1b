"""Controlled generation for Rej-1B."""
from .first import DEFAULT_MAX_NEW_TOKENS, GenerationOutput, generate
from .second import generate_v2

__all__ = ["DEFAULT_MAX_NEW_TOKENS", "GenerationOutput", "generate", "generate_v2"]
