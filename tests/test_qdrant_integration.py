from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lineage.query import LineageQuery

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, PointStruct, VectorParams
except ImportError:  # pragma: no cover - optional dependency
    QdrantClient = None

from connectors.qdrant import TrackedQdrant


def _build_local_qdrant_client(tmpdir: str):
    errors: list[Exception] = []
    for kwargs in ({"location": ":memory:"}, {"path": str(Path(tmpdir) / "qdrant")}):
        try:
            return QdrantClient(**kwargs)
        except Exception as exc:  # pragma: no cover - version dependent
            errors.append(exc)
    raise RuntimeError(f"Could not initialize local Qdrant client: {errors}")


@unittest.skipIf(QdrantClient is None, "qdrant-client is not installed")
class QdrantIntegrationTests(unittest.TestCase):
    def test_upsert_with_lineage_records_sqlite_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "vectrace.db")
            collection = "support_kb"
            client = _build_local_qdrant_client(td)
            try:
                client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=3, distance=Distance.COSINE),
                )

                with TrackedQdrant(db_path=db_path, client=client) as tracked:
                    points = [
                        PointStruct(
                            id=0,
                            vector=[0.1, 0.2, 0.3],
                            payload={"text": "Customer wants a refund", "chunk_index": 0},
                        ),
                        PointStruct(
                            id=1,
                            vector=[0.4, 0.5, 0.6],
                            payload={"text": "Warranty covers defects", "chunk_index": 1},
                        ),
                    ]
                    tracked.upsert_with_lineage(
                        collection_name=collection,
                        points=points,
                        document_id="doc_123",
                        document_path="s3://bucket/support.pdf",
                        chunk_strategy="semantic",
                        embedding_model="text-embedding-3-small",
                        model_version="2024-06-01",
                    )

                with LineageQuery(db_path) as query:
                    lineage = query.get_lineage("0", collection)
                    self.assertIsNotNone(lineage)
                    if lineage is None:
                        return
                    self.assertEqual(lineage["document"]["source_path"], "s3://bucket/support.pdf")
                    self.assertEqual(lineage["chunk"]["strategy"], "semantic")
                    self.assertEqual(lineage["vector"]["embedding_model"], "text-embedding-3-small")
            finally:
                client.close()


if __name__ == "__main__":
    unittest.main()
