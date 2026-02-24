"""
Environment configuration utilities.

Loads values from `.env` for local development while preserving real
environment variables in production (Modal, containers, CI).
"""

from __future__ import annotations

from dotenv import load_dotenv


def load_environment() -> None:
    load_dotenv(override=False)

