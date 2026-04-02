from __future__ import annotations

import sqlite3
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import connectors.qdrant as qdrant_module
from connectors.qdrant import TrackedQdrant
from lineage.query import LineageQuery


@dataclass
class FakePoint:
    id: int
    vector: list[float]
    payload: dict


class FakeClient:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.closed = False
        self.upsert_calls = 0
        self.delete_calls = 0
        self.deleted_payloads: list[tuple[str, list]] = []

    def upsert(self, collection_name: str, points: list) -> None:
        self.upsert_calls += 1
        if self.fail:
            raise RuntimeError("upsert failed")

    def delete(self, collection_name: str, points_selector: list) -> None:
        self.delete_calls += 1
        self.deleted_payloads.append((collection_name, list(points_selector)))

    def close(self) -> None:
        self.closed = True


class LegacyPositionalDeleteClient(FakeClient):
    def delete(self, collection_name: str, selector: list) -> None:  # pragma: no cover - behavior test
        self.delete_calls += 1
        self.deleted_payloads.append((collection_name, list(selector)))


class SelectorOnlyDeleteClient(FakeClient):
    def delete(self, collection_name: str, points_selector) -> None:  # pragma: no cover - behavior test
        if isinstance(points_selector, list):
            raise TypeError("expects selector object")
        points = getattr(points_selector, "points", None)
        if points is None:
            raise TypeError("missing points")
        self.delete_calls += 1
        self.deleted_payloads.append((collection_name, list(points)))


class QdrantConnectorUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "vectrace.db")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_injected_client_is_not_closed_by_connector(self) -> None:
        client = FakeClient()
        tracked = TrackedQdrant(db_path=self.db_path, client=client)
        tracked.close()
        self.assertFalse(client.closed)

    def test_upsert_failure_marks_pipeline_failed_without_lineage_rows(self) -> None:
        client = FakeClient(fail=True)
        tracked = TrackedQdrant(db_path=self.db_path, client=client)
        points = [FakePoint(id=1, vector=[0.1, 0.2, 0.3], payload={"text": "x", "chunk_index": 0})]

        with self.assertRaises(RuntimeError):
            tracked.upsert_with_lineage(
                collection_name="support_kb",
                points=points,
                document_id="doc_1",
                document_path="/tmp/doc.txt",
                embedding_model="m1",
            )

        conn = sqlite3.connect(self.db_path)
        try:
            runs = conn.execute("SELECT status FROM pipeline_runs").fetchall()
            docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            vectors = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        finally:
            conn.close()
            tracked.close()

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0][0], "failed")
        self.assertEqual(docs, 0)
        self.assertEqual(chunks, 0)
        self.assertEqual(vectors, 0)

    def test_upsert_success_records_lineage(self) -> None:
        client = FakeClient(fail=False)
        with TrackedQdrant(db_path=self.db_path, client=client) as tracked:
            points = [
                FakePoint(id=0, vector=[0.1, 0.2, 0.3], payload={"text": "hello", "chunk_index": 0}),
                FakePoint(id=1, vector=[0.4, 0.5, 0.6], payload={"text": "world", "chunk_index": 1}),
            ]
            tracked.upsert_with_lineage(
                collection_name="support_kb",
                points=points,
                document_id="doc_1",
                document_path="s3://bucket/doc.pdf",
                chunk_strategy="semantic",
                embedding_model="text-embedding-3-small",
                model_version="2024-06-01",
            )

        self.assertEqual(client.upsert_calls, 1)
        with LineageQuery(self.db_path) as query:
            lineage = query.get_lineage("0", "support_kb")
            self.assertIsNotNone(lineage)
            if lineage is None:
                return
            self.assertEqual(lineage["document"]["source_type"], "s3")
            self.assertEqual(lineage["chunk"]["strategy"], "semantic")
            self.assertEqual(lineage["vector"]["model_version"], "2024-06-01")

    def test_lineage_write_failure_triggers_best_effort_qdrant_rollback(self) -> None:
        client = FakeClient(fail=False)
        tracked = TrackedQdrant(db_path=self.db_path, client=client)
        points = [
            FakePoint(id=0, vector=[0.1, 0.2, 0.3], payload={"text": "t0", "chunk_index": 0}),
            FakePoint(id=1, vector=[0.4, 0.5, 0.6], payload={"text": "t1", "chunk_index": 1}),
        ]
        with unittest.mock.patch.object(
            tracked.tracker, "record_vectors", side_effect=RuntimeError("db write failed")
        ):
            with self.assertRaises(RuntimeError):
                tracked.upsert_with_lineage(
                    collection_name="support_kb",
                    points=points,
                    document_id="doc_1",
                    document_path="/tmp/doc.txt",
                    embedding_model="m1",
                )
        tracked.close()

        self.assertEqual(client.upsert_calls, 1)
        self.assertEqual(client.delete_calls, 1)
        self.assertEqual(client.deleted_payloads[0], ("support_kb", [0, 1]))

    def test_rollback_supports_legacy_positional_delete_signature(self) -> None:
        client = LegacyPositionalDeleteClient(fail=False)
        tracked = TrackedQdrant(db_path=self.db_path, client=client)
        points = [
            FakePoint(id=11, vector=[0.1, 0.2, 0.3], payload={"text": "a", "chunk_index": 0}),
            FakePoint(id=12, vector=[0.4, 0.5, 0.6], payload={"text": "b", "chunk_index": 1}),
        ]
        with unittest.mock.patch.object(
            tracked.tracker, "record_vectors", side_effect=RuntimeError("db write failed")
        ):
            with self.assertRaises(RuntimeError):
                tracked.upsert_with_lineage(
                    collection_name="support_kb",
                    points=points,
                    document_id="doc_legacy",
                    document_path="/tmp/doc.txt",
                    embedding_model="m1",
                )
        tracked.close()

        self.assertEqual(client.delete_calls, 1)
        self.assertEqual(client.deleted_payloads[0], ("support_kb", [11, 12]))

    def test_rollback_supports_pointidslist_selector(self) -> None:
        client = SelectorOnlyDeleteClient(fail=False)
        tracked = TrackedQdrant(db_path=self.db_path, client=client)
        points = [
            FakePoint(id=21, vector=[0.1, 0.2, 0.3], payload={"text": "a", "chunk_index": 0}),
            FakePoint(id=22, vector=[0.4, 0.5, 0.6], payload={"text": "b", "chunk_index": 1}),
        ]

        class _PointIdsList:
            def __init__(self, points: list):
                self.points = points

        class _Models:
            PointIdsList = _PointIdsList

        with unittest.mock.patch.object(qdrant_module, "qdrant_models", _Models):
            with unittest.mock.patch.object(
                tracked.tracker, "record_vectors", side_effect=RuntimeError("db write failed")
            ):
                with self.assertRaises(RuntimeError):
                    tracked.upsert_with_lineage(
                        collection_name="support_kb",
                        points=points,
                        document_id="doc_selector",
                        document_path="/tmp/doc.txt",
                        embedding_model="m1",
                    )
        tracked.close()

        self.assertEqual(client.delete_calls, 1)
        self.assertEqual(client.deleted_payloads[0], ("support_kb", [21, 22]))


if __name__ == "__main__":
    unittest.main()
