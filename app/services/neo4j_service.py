"""
Neo4j connection manager — singleton driver shared across the app.

On Modal, credentials come from modal.Secret("neo4j-credentials")
which sets NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD as env vars.
"""

from __future__ import annotations
import logging
import os
from typing import Any

from neo4j import GraphDatabase, Driver

from app.config import load_environment

load_environment()

log = logging.getLogger(__name__)

_driver: Driver | None = None


def _get_neo4j_config() -> tuple[str, str, str]:
    """Read Neo4j config from environment (set by Modal secrets)."""
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")
    if not password:
        log.warning("NEO4J_PASSWORD is not set; Neo4j connectivity may fail.")
    return uri, user, password


def get_driver() -> Driver:
    """Return (and lazily create) the Neo4j driver singleton."""
    global _driver
    if _driver is None:
        uri, user, password = _get_neo4j_config()
        _driver = GraphDatabase.driver(uri, auth=(user, password))
        log.info("Neo4j driver created → %s", uri)
    return _driver


def close_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
        log.info("Neo4j driver closed")


def cypher(query: str, **params: Any) -> list[dict]:
    """Run a Cypher query and return list of dicts."""
    with get_driver().session() as session:
        return session.run(query, **params).data()


def check_connection() -> bool:
    """Return True if Neo4j is reachable."""
    try:
        get_driver().verify_connectivity()
        return True
    except Exception:
        return False
