from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lineage.models import create_connection
from lineage.query import AmbiguousVectorIDError, LineageQuery
from lineage.tracker import LineageTracker


class TrackerQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "vectrace.db")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_record_and_query_lineage(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("unit_test")
            tracker.record_document(
                doc_id="doc_1",
                source_path="s3://bucket/doc.pdf",
                source_type="s3",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_1:chunk:0",
                document_id="doc_1",
                chunk_index=0,
                strategy="semantic",
                chunk_size=21,
                text_preview="refund for broken item",
            )
            tracker.record_vector(
                vector_id="0",
                collection_name="support_kb",
                chunk_id="doc_1:chunk:0",
                embedding_model="text-embedding-3-small",
                model_version="2024-06-01",
            )
            tracker.complete_pipeline("success")

        with LineageQuery(self.db_path) as query:
            lineage = query.get_lineage("0", "support_kb")

        self.assertIsNotNone(lineage)
        if lineage is None:
            return
        self.assertEqual(lineage["vector"]["id"], "0")
        self.assertEqual(lineage["vector"]["collection_name"], "support_kb")
        self.assertEqual(lineage["chunk"]["id"], "doc_1:chunk:0")
        self.assertEqual(lineage["document"]["id"], "doc_1")

    def test_ambiguous_vector_id_requires_collection(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("unit_test")
            tracker.record_document(
                doc_id="doc_1",
                source_path="/tmp/doc.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_1:chunk:0",
                document_id="doc_1",
                chunk_index=0,
                strategy="fixed-size",
                chunk_size=3,
                text_preview="abc",
            )
            tracker.record_vector(
                vector_id="same-id",
                collection_name="collection_a",
                chunk_id="doc_1:chunk:0",
                embedding_model="model_a",
            )
            tracker.record_vector(
                vector_id="same-id",
                collection_name="collection_b",
                chunk_id="doc_1:chunk:0",
                embedding_model="model_b",
            )
            tracker.complete_pipeline("success")

        with LineageQuery(self.db_path) as query:
            with self.assertRaises(AmbiguousVectorIDError):
                query.get_lineage("same-id")

    def test_foreign_keys_enforced(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            with self.assertRaises(sqlite3.IntegrityError):
                tracker.record_vector(
                    vector_id="v1",
                    collection_name="support_kb",
                    chunk_id="missing_chunk",
                    embedding_model="model_x",
                )

    def test_batch_id_rotates_per_pipeline_run(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("run_a")
            batch_a = tracker.batch_id
            tracker.complete_pipeline("success")

            tracker.start_pipeline("run_b")
            batch_b = tracker.batch_id
            tracker.complete_pipeline("success")

        self.assertNotEqual(batch_a, batch_b)

    def test_start_pipeline_rejects_overlapping_run(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("run_a")
            with self.assertRaises(RuntimeError):
                tracker.start_pipeline("run_b")

    def test_context_manager_marks_run_failed_on_exception(self) -> None:
        with self.assertRaises(RuntimeError):
            with LineageTracker(self.db_path, autoinit=True) as tracker:
                run_id = tracker.start_pipeline("run_will_fail")
                self.assertIsNotNone(run_id)
                raise RuntimeError("boom")

        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT status, completed_at FROM pipeline_runs WHERE name = ?",
                ("run_will_fail",),
            ).fetchone()
        finally:
            conn.close()

        self.assertIsNotNone(row)
        if row is None:
            return
        self.assertEqual(row[0], "failed")
        self.assertIsNotNone(row[1])

    def test_record_vector_upsert_updates_created_timestamp(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("upsert_timestamp")
            tracker.record_document(
                doc_id="doc_upsert",
                source_path="/tmp/doc-upsert.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_upsert:chunk:0",
                document_id="doc_upsert",
                chunk_index=0,
                strategy="semantic",
                chunk_size=12,
                text_preview="first chunk",
            )

            with patch("lineage.tracker._utc_now_iso", return_value="2026-04-01T00:00:00+00:00"):
                tracker.record_vector(
                    vector_id="v_upsert",
                    collection_name="support_kb",
                    chunk_id="doc_upsert:chunk:0",
                    embedding_model="model_v1",
                    model_version="1",
                )

            with patch("lineage.tracker._utc_now_iso", return_value="2026-04-01T00:00:05+00:00"):
                tracker.record_vector(
                    vector_id="v_upsert",
                    collection_name="support_kb",
                    chunk_id="doc_upsert:chunk:0",
                    embedding_model="model_v2",
                    model_version="2",
                )
            tracker.complete_pipeline("success")

        with LineageQuery(self.db_path) as query:
            lineage = query.get_lineage("v_upsert", "support_kb")
        self.assertIsNotNone(lineage)
        if lineage is None:
            return
        self.assertEqual(lineage["vector"]["embedding_model"], "model_v2")
        self.assertEqual(lineage["vector"]["model_version"], "2")
        self.assertEqual(lineage["vector"]["created_at"], "2026-04-01T00:00:05+00:00")

    def test_get_latest_retrieval_event_for_vector(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("retrieval_lookup")
            tracker.record_document(
                doc_id="doc_ret",
                source_path="/tmp/doc_ret.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_ret:chunk:0",
                document_id="doc_ret",
                chunk_index=0,
                strategy="semantic",
                chunk_size=9,
                text_preview="ret chunk",
            )
            tracker.record_vector(
                vector_id="ret_v",
                collection_name="support_kb",
                chunk_id="doc_ret:chunk:0",
                embedding_model="m1",
            )
            tracker.record_retrieval_event(
                query_id="query_a",
                query_text="first query",
                final_answer="a1",
                collection_name="support_kb",
                vector_id="ret_v",
                rank=2,
                score=0.3,
                metadata_json='{"k":"v1"}',
                event_id="event_a",
            )
            tracker.record_retrieval_event(
                query_id="query_b",
                query_text="latest query",
                final_answer="a2",
                collection_name="support_kb",
                vector_id="ret_v",
                rank=1,
                score=0.9,
                metadata_json='{"k":"v2"}',
                event_id="event_b",
            )
            tracker.complete_pipeline("success")

        with LineageQuery(self.db_path) as query:
            event = query.get_latest_retrieval_event("ret_v", "support_kb")
        self.assertIsNotNone(event)
        if event is None:
            return
        self.assertEqual(event["id"], "event_b")
        self.assertEqual(event["query_text"], "latest query")
        self.assertEqual(event["metadata"]["k"], "v2")

    def test_find_retrieval_events_filters(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("retrieval_find")
            tracker.record_document(
                doc_id="doc_find",
                source_path="/tmp/doc_find.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_find:chunk:0",
                document_id="doc_find",
                chunk_index=0,
                strategy="semantic",
                chunk_size=5,
                text_preview="find",
            )
            tracker.record_vector(
                vector_id="find_v1",
                collection_name="support_kb",
                chunk_id="doc_find:chunk:0",
                embedding_model="m1",
            )
            tracker.record_retrieval_event(
                query_id="q1",
                query_text="what is refund policy",
                final_answer="answer a",
                collection_name="support_kb",
                vector_id="find_v1",
                rank=1,
                score=0.5,
                event_id="ev_find_1",
            )
            tracker.record_retrieval_event(
                query_id="q2",
                query_text="what is refund policy",
                final_answer="answer b",
                collection_name="support_kb",
                vector_id="find_v1",
                rank=2,
                score=0.4,
                event_id="ev_find_2",
            )
            tracker.complete_pipeline("success")

        with LineageQuery(self.db_path) as query:
            all_events = query.find_retrieval_events("what is refund policy", limit=10)
            only_answer_a = query.find_retrieval_events(
                "what is refund policy", final_answer="answer a", limit=10
            )

        self.assertEqual(len(all_events), 2)
        self.assertEqual(len(only_answer_a), 1)
        self.assertEqual(only_answer_a[0]["id"], "ev_find_1")

    def test_find_trace_candidates_ranks_by_overlap(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("trace_candidates")
            tracker.record_document(
                doc_id="doc_candidates",
                source_path="/tmp/doc_candidates.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_candidates:chunk:refund",
                document_id="doc_candidates",
                chunk_index=0,
                strategy="semantic",
                chunk_size=32,
                text_preview="Refund policy allows returns within 30 days.",
            )
            tracker.record_chunk(
                chunk_id="doc_candidates:chunk:warranty",
                document_id="doc_candidates",
                chunk_index=1,
                strategy="semantic",
                chunk_size=28,
                text_preview="Warranty terms for manufacturer defects.",
            )
            tracker.record_vector(
                vector_id="cand_refund",
                collection_name="support_kb",
                chunk_id="doc_candidates:chunk:refund",
                embedding_model="m1",
            )
            tracker.record_vector(
                vector_id="cand_warranty",
                collection_name="support_kb",
                chunk_id="doc_candidates:chunk:warranty",
                embedding_model="m1",
            )
            tracker.complete_pipeline("success")

        with LineageQuery(self.db_path) as query:
            candidates = query.find_trace_candidates(
                question="Can I get a refund after 90 days?",
                collection_name="support_kb",
                limit=5,
            )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["vector_id"], "cand_refund")
        self.assertGreater(candidates[0]["score"], 0.0)
        self.assertIn("refund", candidates[0]["overlap_terms"])

    def test_find_trace_candidates_is_deterministic_on_ties(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("trace_candidates_ties")
            tracker.record_document(
                doc_id="doc_candidates_ties",
                source_path="/tmp/doc_candidates_ties.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_candidates_ties:chunk:2",
                document_id="doc_candidates_ties",
                chunk_index=2,
                strategy="semantic",
                chunk_size=20,
                text_preview="refund policy terms",
            )
            tracker.record_chunk(
                chunk_id="doc_candidates_ties:chunk:1",
                document_id="doc_candidates_ties",
                chunk_index=1,
                strategy="semantic",
                chunk_size=20,
                text_preview="refund policy terms",
            )
            tracker.record_vector(
                vector_id="cand_tie_b",
                collection_name="support_kb",
                chunk_id="doc_candidates_ties:chunk:2",
                embedding_model="m1",
            )
            tracker.record_vector(
                vector_id="cand_tie_a",
                collection_name="support_kb",
                chunk_id="doc_candidates_ties:chunk:1",
                embedding_model="m1",
            )
            tracker.complete_pipeline("success")

        with LineageQuery(self.db_path) as query:
            candidates = query.find_trace_candidates(
                question="refund policy",
                collection_name="support_kb",
                limit=2,
            )

        self.assertEqual([c["vector_id"] for c in candidates], ["cand_tie_a", "cand_tie_b"])

    def test_get_retrieval_event_by_id(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("retrieval_by_id")
            tracker.record_document(
                doc_id="doc_by_id",
                source_path="/tmp/doc_by_id.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_by_id:chunk:0",
                document_id="doc_by_id",
                chunk_index=0,
                strategy="semantic",
                chunk_size=5,
                text_preview="hello",
            )
            tracker.record_vector(
                vector_id="by_id_v",
                collection_name="support_kb",
                chunk_id="doc_by_id:chunk:0",
                embedding_model="m1",
            )
            tracker.record_retrieval_event(
                query_id="q_by_id",
                query_text="where",
                final_answer="there",
                collection_name="support_kb",
                vector_id="by_id_v",
                rank=1,
                score=0.8,
                event_id="event_by_id",
            )
            tracker.complete_pipeline("success")

        with LineageQuery(self.db_path) as query:
            event = query.get_retrieval_event("event_by_id")

        self.assertIsNotNone(event)
        if event is None:
            return
        self.assertEqual(event["query_id"], "q_by_id")
        self.assertEqual(event["vector_id"], "by_id_v")


    def test_apply_idempotent_alters_propagates_non_duplicate_errors(self) -> None:
        from lineage.models import _apply_idempotent_alters

        class FakeConn:
            def execute(self, sql: str) -> None:
                raise sqlite3.OperationalError("database is locked")

        with self.assertRaises(sqlite3.OperationalError):
            _apply_idempotent_alters(FakeConn())  # type: ignore[arg-type]

    def test_apply_idempotent_alters_swallows_duplicate_column(self) -> None:
        from lineage.models import _apply_idempotent_alters

        class FakeConn:
            def execute(self, sql: str) -> None:
                raise sqlite3.OperationalError("duplicate column name: source_url")

        # Must NOT raise — that is the intended idempotent path.
        _apply_idempotent_alters(FakeConn())  # type: ignore[arg-type]

    def test_create_connection_sets_busy_timeout(self) -> None:
        conn = create_connection(self.db_path)
        try:
            row = conn.execute("PRAGMA busy_timeout").fetchone()
        finally:
            conn.close()
        self.assertIsNotNone(row)
        if row is None:
            return
        self.assertEqual(int(row[0]), 30000)

    def test_document_deep_link_fields_round_trip(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("unit_test")
            tracker.record_document(
                doc_id="doc_dl",
                source_path="s3://bucket/policy.pdf",
                source_type="s3",
                version="v1",
                source_url="https://example.com/policy.pdf",
                source_page=7,
                source_section="3.2 Refunds",
            )
            tracker.record_chunk(
                chunk_id="doc_dl:chunk:0",
                document_id="doc_dl",
                chunk_index=0,
                strategy="semantic",
                chunk_size=12,
                text_preview="refund text",
            )
            tracker.record_vector(
                vector_id="vec_dl",
                collection_name="support_kb",
                chunk_id="doc_dl:chunk:0",
                embedding_model="m1",
            )
            tracker.complete_pipeline("success")

        with LineageQuery(self.db_path) as query:
            lineage = query.get_lineage("vec_dl", "support_kb")

        self.assertIsNotNone(lineage)
        if lineage is None:
            return
        self.assertEqual(lineage["document"]["source_url"], "https://example.com/policy.pdf")
        self.assertEqual(lineage["document"]["source_page"], 7)
        self.assertEqual(lineage["document"]["source_section"], "3.2 Refunds")


if __name__ == "__main__":
    unittest.main()
