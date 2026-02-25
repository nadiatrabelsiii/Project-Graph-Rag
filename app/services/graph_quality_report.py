#!/usr/bin/env python3
"""Graph quality report for Neo4j legal Graph RAG backend.

Usage:
  PYTHONPATH=. python3 app/services/graph_quality_report.py \
    --neo4j-uri "$NEO4J_URI" --neo4j-user "$NEO4J_USER" --neo4j-password "$NEO4J_PASSWORD" \
    --neo4j-database "$NEO4J_DATABASE"
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from app.config import load_environment

load_environment()


def _print_table(title: str, rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    if not rows:
        print("(no rows)")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  " + " | ".join(c.ljust(widths[c]) for c in cols))
    print("  " + "-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  " + " | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def run_report(uri: str, user: str, password: str, database: str | None = None) -> None:
    from neo4j import GraphDatabase

    drv = GraphDatabase.driver(uri, auth=(user, password))
    try:
        session_kwargs: dict[str, Any] = {"database": database} if database else {}
        with drv.session(**session_kwargs) as s:
            q = s.run

            _print_table(
                "Node Counts",
                q(
                    "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY count DESC"
                ).data(),
            )

            _print_table(
                "Relationship Counts",
                q(
                    "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS count ORDER BY count DESC"
                ).data(),
            )

            _print_table(
                "Chunk Coverage by Source/Year/Type",
                q(
                    "MATCH (c:Chunk) "
                    "RETURN c.source AS source, c.document_year AS year, c.chunk_type AS type, count(*) AS count "
                    "ORDER BY source, year, type"
                ).data(),
            )

            _print_table(
                "Orphan Chunk Checks",
                q(
                    "MATCH (c:Chunk) "
                    "WITH c, "
                    "  EXISTS((c)-[:IN_DOCUMENT]->(:Document)) AS has_doc, "
                    "  EXISTS((c)-[:NEXT_CHUNK]->(:Chunk)) OR EXISTS((:Chunk)-[:NEXT_CHUNK]->(c)) AS has_seq "
                    "RETURN "
                    "  sum(CASE WHEN has_doc THEN 0 ELSE 1 END) AS chunks_without_document, "
                    "  sum(CASE WHEN has_seq THEN 0 ELSE 1 END) AS chunks_without_sequence"
                ).data(),
            )

            _print_table(
                "Year Consistency Checks",
                q(
                    "MATCH (n:Chunk)-[r:EXPLAINS|ABOUT_ARTICLE|RELATES_TO]->(l:Chunk) "
                    "WITH type(r) AS rel, count(*) AS total, "
                    "sum(CASE WHEN n.document_year = l.document_year THEN 0 ELSE 1 END) AS year_mismatch "
                    "RETURN rel, total, year_mismatch ORDER BY total DESC"
                ).data(),
            )

            _print_table(
                "Top EXPLAINS Fanout (potentially noisy notes)",
                q(
                    "MATCH (n:Chunk {source:'notes'})-[r:EXPLAINS]->(l:Chunk {source:'law'}) "
                    "RETURN n.source_file AS note_file, n.section_path AS section, count(r) AS explains_links "
                    "ORDER BY explains_links DESC LIMIT 20"
                ).data(),
            )

            _print_table(
                "Top RELATES_TO Fanout (potentially dense links)",
                q(
                    "MATCH (n:Chunk {source:'notes'})-[r:RELATES_TO]->(l:Chunk {source:'law'}) "
                    "RETURN n.source_file AS note_file, n.section_path AS section, count(r) AS relates_links "
                    "ORDER BY relates_links DESC LIMIT 20"
                ).data(),
            )

            _print_table(
                "ABOUT_ARTICLE Precision Signals",
                q(
                    "MATCH (n:Chunk {source:'notes'})-[r:ABOUT_ARTICLE]->(l:Chunk {source:'law'}) "
                    "RETURN "
                    "  count(r) AS total_links, "
                    "  sum(CASE WHEN coalesce(r.explicit_article_ref, false) THEN 1 ELSE 0 END) AS explicit_links, "
                    "  avg(coalesce(r.topic_overlap, 0)) AS avg_topic_overlap"
                ).data(),
            )

            _print_table(
                "CROSS_REFERENCES Quality",
                q(
                    "MATCH (a:Chunk)-[r:CROSS_REFERENCES]->(b:Chunk) "
                    "RETURN "
                    "  count(r) AS total_edges, "
                    "  sum(CASE WHEN a.document_year = b.document_year THEN 1 ELSE 0 END) AS same_year_edges, "
                    "  avg(coalesce(r.mentions, 1)) AS avg_mentions_per_edge"
                ).data(),
            )

            _print_table(
                "External Article Citation Quality",
                q(
                    "MATCH (c:Chunk)-[r:CITES_EXTERNAL_ARTICLE]->(x:ExternalArticle) "
                    "RETURN "
                    "  count(r) AS total_edges, "
                    "  count(DISTINCT x.external_id) AS external_article_nodes, "
                    "  avg(coalesce(r.mentions, 1)) AS avg_mentions_per_edge"
                ).data(),
            )

            _print_table(
                "Top External Law Years Referenced",
                q(
                    "MATCH (c:Chunk)-[:CITES_EXTERNAL_ARTICLE]->(x:ExternalArticle) "
                    "RETURN x.source_year AS source_year, count(*) AS citations "
                    "ORDER BY citations DESC LIMIT 15"
                ).data(),
            )

            _print_table(
                "Topic Coverage",
                q(
                    "MATCH (c:Chunk) "
                    "OPTIONAL MATCH (c)-[:HAS_TOPIC]->(t:Topic) "
                    "WITH c, count(t) AS tcount "
                    "RETURN c.source AS source, "
                    "       count(*) AS chunks, "
                    "       sum(CASE WHEN tcount > 0 THEN 1 ELSE 0 END) AS chunks_with_topics, "
                    "       round(100.0 * sum(CASE WHEN tcount > 0 THEN 1 ELSE 0 END) / count(*), 2) AS pct_with_topics"
                    " ORDER BY source"
                ).data(),
            )

    finally:
        drv.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Neo4j Graph RAG quality checks")
    ap.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    ap.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME", "neo4j"))
    ap.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", ""))
    ap.add_argument("--neo4j-database", default=os.environ.get("NEO4J_DATABASE", ""))
    args = ap.parse_args()

    if not args.neo4j_password:
        raise SystemExit("Missing Neo4j password. Set NEO4J_PASSWORD or pass --neo4j-password.")

    run_report(args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.neo4j_database or None)


if __name__ == "__main__":
    main()
