"""Seed sample lineage records into a local VecTrace database."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lineage.tracker import LineageTracker


def main() -> None:
    with LineageTracker("./vectrace.db", autoinit=True) as tracker:
        tracker.start_pipeline("demo_seed")
        tracker.record_document(
            doc_id="doc_123",
            source_path="s3://bucket/customer_support_2024.pdf",
            source_type="s3",
            version="v1",
            content_hash="sha256:example",
        )
        tracker.record_chunk(
            chunk_id="doc_123:chunk:0",
            document_id="doc_123",
            chunk_index=0,
            strategy="semantic",
            chunk_size=47,
            text_preview="Customer wants a refund for the broken product",
        )
        tracker.record_vector(
            vector_id="0",
            collection_name="support_kb",
            chunk_id="doc_123:chunk:0",
            embedding_model="text-embedding-3-small",
            model_version="2024-06-01",
        )
        tracker.complete_pipeline("success")
    print("Seeded demo lineage into ./vectrace.db")


if __name__ == "__main__":
    main()
