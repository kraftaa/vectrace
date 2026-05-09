from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from lineage.tracker import LineageTracker
from vectrace import cli


class CLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "vectrace.db")
        self.report_path = str(Path(self.tmp.name) / "lineage.html")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_init_creates_tables(self) -> None:
        result = cli(["init", "--db", self.db_path])
        self.assertEqual(result, 0)

        conn = sqlite3.connect(self.db_path)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        finally:
            conn.close()

        self.assertTrue({"documents", "chunks", "vectors", "pipeline_runs"}.issubset(tables))

    def test_onboard_seeds_and_generates_report(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = cli(
                [
                    "onboard",
                    "--db",
                    self.db_path,
                    "--vector-id",
                    "vec_demo_test",
                    "--collection",
                    "support_kb",
                    "--output",
                    self.report_path,
                ]
            )
        self.assertEqual(result, 0)
        self.assertTrue(Path(self.report_path).exists())

        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM vectors WHERE id = ? AND collection_name = ?",
                ("vec_demo_test", "support_kb"),
            ).fetchone()
        finally:
            conn.close()
        self.assertIsNotNone(row)
        if row is None:
            return
        self.assertEqual(row[0], 1)
        self.assertIn("Generated demo report", stdout.getvalue())

    def test_onboard_skip_seed(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = cli(["onboard", "--db", self.db_path, "--skip-seed"])
        self.assertEqual(result, 0)
        self.assertIn("Skipped demo seed", stdout.getvalue())

        conn = sqlite3.connect(self.db_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(count, 0)

    def test_onboard_multiple_vector_ids_do_not_collide(self) -> None:
        result_a = cli(
            [
                "onboard",
                "--db",
                self.db_path,
                "--vector-id",
                "vec_demo_a",
                "--collection",
                "support_kb",
                "--output",
                self.report_path,
            ]
        )
        self.assertEqual(result_a, 0)

        result_b = cli(
            [
                "onboard",
                "--db",
                self.db_path,
                "--vector-id",
                "vec_demo_b",
                "--collection",
                "support_kb",
                "--output",
                self.report_path,
            ]
        )
        self.assertEqual(result_b, 0)

        conn = sqlite3.connect(self.db_path)
        try:
            vector_count = conn.execute(
                "SELECT COUNT(*) FROM vectors WHERE collection_name = ?",
                ("support_kb",),
            ).fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        finally:
            conn.close()

        self.assertEqual(vector_count, 2)
        self.assertGreaterEqual(chunk_count, 2)

    def test_seed_demo_appends_vectors(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = cli(
                [
                    "seed-demo",
                    "--db",
                    self.db_path,
                    "--collection",
                    "support_kb",
                    "--vectors",
                    "7",
                    "--docs",
                    "3",
                    "--start-index",
                    "10",
                    "--prefix",
                    "vec_test_",
                ]
            )
        self.assertEqual(result, 0)
        output = stdout.getvalue()
        self.assertIn("Seeded 7 synthetic vectors", output)
        self.assertIn("vec_test_00010", output)

        conn = sqlite3.connect(self.db_path)
        try:
            total = conn.execute(
                "SELECT COUNT(*) FROM vectors WHERE collection_name = ?",
                ("support_kb",),
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(total, 7)

    def test_seed_demo_validates_inputs(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = cli(["seed-demo", "--db", self.db_path, "--vectors", "0"])
        self.assertEqual(result, 2)
        self.assertIn("--vectors must be > 0", stderr.getvalue())

    def test_seed_demo_validates_start_index(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = cli(["seed-demo", "--db", self.db_path, "--start-index", "-1"])
        self.assertEqual(result, 2)
        self.assertIn("--start-index must be >= 0", stderr.getvalue())

    def test_seed_demo_different_prefixes_append_without_chunk_collisions(self) -> None:
        result_a = cli(
            [
                "seed-demo",
                "--db",
                self.db_path,
                "--collection",
                "support_kb",
                "--vectors",
                "3",
                "--docs",
                "1",
                "--start-index",
                "0",
                "--prefix",
                "vec_a_",
            ]
        )
        self.assertEqual(result_a, 0)

        result_b = cli(
            [
                "seed-demo",
                "--db",
                self.db_path,
                "--collection",
                "support_kb",
                "--vectors",
                "3",
                "--docs",
                "1",
                "--start-index",
                "0",
                "--prefix",
                "vec_b_",
            ]
        )
        self.assertEqual(result_b, 0)

        conn = sqlite3.connect(self.db_path)
        try:
            vector_count = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        finally:
            conn.close()

        self.assertEqual(vector_count, 6)
        self.assertEqual(chunk_count, 6)

    def test_seed_demo_auto_start_index_appends_on_repeated_runs(self) -> None:
        first = cli(
            [
                "seed-demo",
                "--db",
                self.db_path,
                "--collection",
                "support_kb",
                "--vectors",
                "2",
                "--docs",
                "1",
                "--prefix",
                "vec_auto_",
            ]
        )
        self.assertEqual(first, 0)

        second_stdout = io.StringIO()
        with redirect_stdout(second_stdout):
            second = cli(
                [
                    "seed-demo",
                    "--db",
                    self.db_path,
                    "--collection",
                    "support_kb",
                    "--vectors",
                    "2",
                    "--docs",
                    "1",
                    "--prefix",
                    "vec_auto_",
                ]
            )
        self.assertEqual(second, 0)

        conn = sqlite3.connect(self.db_path)
        try:
            vector_count = conn.execute(
                "SELECT COUNT(*) FROM vectors WHERE id LIKE 'vec_auto_%' AND collection_name = ?",
                ("support_kb",),
            ).fetchone()[0]
        finally:
            conn.close()

        self.assertEqual(vector_count, 4)
        self.assertIn("vec_auto_00002", second_stdout.getvalue())

    def test_seed_demo_surfaces_infer_start_index_db_errors(self) -> None:
        with patch("vectrace.infer_next_start_index", side_effect=sqlite3.OperationalError("db locked")):
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = cli(
                    [
                        "seed-demo",
                        "--db",
                        self.db_path,
                        "--collection",
                        "support_kb",
                        "--vectors",
                        "2",
                        "--docs",
                        "1",
                    ]
                )
        self.assertEqual(result, 2)
        self.assertIn("Trace database error", stderr.getvalue())

    def test_onboard_surfaces_vector_exists_db_errors(self) -> None:
        with patch("vectrace.vector_exists", side_effect=sqlite3.OperationalError("db locked")):
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = cli(
                    [
                        "onboard",
                        "--db",
                        self.db_path,
                        "--vector-id",
                        "vec_demo_test",
                        "--collection",
                        "support_kb",
                        "--output",
                        self.report_path,
                    ]
                )
        self.assertEqual(result, 2)
        self.assertIn("Trace database error", stderr.getvalue())

    def test_record_retrieval_surfaces_vector_exists_db_errors(self) -> None:
        with patch("vectrace.vector_exists", side_effect=sqlite3.OperationalError("db locked")):
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = cli(
                    [
                        "record-retrieval",
                        "--db",
                        self.db_path,
                        "--collection",
                        "support_kb",
                        "--vector-id",
                        "v1",
                        "--query-text",
                        "hello",
                    ]
                )
        self.assertEqual(result, 2)
        self.assertIn("Trace database error", stderr.getvalue())

    def test_trace_returns_not_found(self) -> None:
        cli(["init", "--db", self.db_path])
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = cli(["trace", "--vector-id", "missing", "--db", self.db_path])
        self.assertEqual(result, 1)
        self.assertIn("No trace found", stderr.getvalue())

    def test_trace_handles_uninitialized_database(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = cli(["trace", "--vector-id", "missing", "--db", self.db_path])
        self.assertEqual(result, 2)
        self.assertIn("Run `vectrace init --db", stderr.getvalue())

    def test_report_writes_html(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("cli_test")
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
                strategy="semantic",
                chunk_size=4,
                text_preview="text",
            )
            tracker.record_vector(
                vector_id="1",
                collection_name="support_kb",
                chunk_id="doc_1:chunk:0",
                embedding_model="m1",
            )
            tracker.complete_pipeline("success")

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = cli(
                [
                    "report",
                    "--vector-id",
                    "1",
                    "--collection",
                    "support_kb",
                    "--db",
                    self.db_path,
                    "--output",
                    self.report_path,
                ]
            )
        self.assertEqual(result, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertTrue(Path(self.report_path).exists())

    def test_report_handles_uninitialized_database(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = cli(
                [
                    "report",
                    "--vector-id",
                    "1",
                    "--db",
                    self.db_path,
                    "--output",
                    self.report_path,
                ]
            )
        self.assertEqual(result, 2)
        self.assertIn("Run `vectrace init --db", stderr.getvalue())

    def test_trace_json_output(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("cli_json")
            tracker.record_document(
                doc_id="doc_json",
                source_path="/tmp/doc.json",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_json:chunk:0",
                document_id="doc_json",
                chunk_index=0,
                strategy="semantic",
                chunk_size=9,
                text_preview="json-text",
            )
            tracker.record_vector(
                vector_id="j1",
                collection_name="support_kb",
                chunk_id="doc_json:chunk:0",
                embedding_model="m1",
            )
            tracker.complete_pipeline("success")

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = cli(
                [
                    "trace",
                    "--vector-id",
                    "j1",
                    "--collection",
                    "support_kb",
                    "--db",
                    self.db_path,
                    "--format",
                    "json",
                ]
            )
        self.assertEqual(result, 0)
        output = stdout.getvalue()
        self.assertIn('"schema_version": "1.0"', output)
        self.assertIn('"id": "j1"', output)

    def test_record_retrieval_and_trace_include_retrieval_json(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("cli_retrieval")
            tracker.record_document(
                doc_id="doc_retrieval",
                source_path="/tmp/doc_retrieval.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_retrieval:chunk:0",
                document_id="doc_retrieval",
                chunk_index=0,
                strategy="semantic",
                chunk_size=10,
                text_preview="retrieval",
            )
            tracker.record_vector(
                vector_id="rv1",
                collection_name="support_kb",
                chunk_id="doc_retrieval:chunk:0",
                embedding_model="m1",
            )
            tracker.complete_pipeline("success")

        rec = cli(
            [
                "record-retrieval",
                "--db",
                self.db_path,
                "--collection",
                "support_kb",
                "--vector-id",
                "rv1",
                "--query-text",
                "refund after 90 days",
                "--final-answer",
                "Yes",
                "--rank",
                "1",
                "--score",
                "0.87",
                "--metadata-json",
                '{"session":"abc"}',
            ]
        )
        self.assertEqual(rec, 0)

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            trace = cli(
                [
                    "trace",
                    "--db",
                    self.db_path,
                    "--vector-id",
                    "rv1",
                    "--collection",
                    "support_kb",
                    "--format",
                    "json",
                    "--include-retrieval",
                ]
            )
        self.assertEqual(trace, 0)
        payload = json.loads(stdout.getvalue())
        self.assertIsNotNone(payload.get("retrieval"))
        self.assertEqual(payload["retrieval"]["query_text"], "refund after 90 days")
        self.assertEqual(payload["retrieval"]["rank"], 1)
        self.assertEqual(payload["retrieval"]["metadata"]["session"], "abc")
        self.assertEqual(payload["retrieval"]["trace_mode"], "exact")

    def test_record_retrieval_rejects_invalid_metadata_json(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("cli_retrieval_bad_json")
            tracker.record_document(
                doc_id="doc_bad_json",
                source_path="/tmp/doc_bad_json.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_bad_json:chunk:0",
                document_id="doc_bad_json",
                chunk_index=0,
                strategy="semantic",
                chunk_size=4,
                text_preview="bad",
            )
            tracker.record_vector(
                vector_id="bad_json_v",
                collection_name="support_kb",
                chunk_id="doc_bad_json:chunk:0",
                embedding_model="m1",
            )
            tracker.complete_pipeline("success")

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = cli(
                [
                    "record-retrieval",
                    "--db",
                    self.db_path,
                    "--collection",
                    "support_kb",
                    "--vector-id",
                    "bad_json_v",
                    "--query-text",
                    "q",
                    "--metadata-json",
                    "{bad",
                ]
            )
        self.assertEqual(result, 2)
        self.assertIn("valid JSON", stderr.getvalue())

    def test_record_retrieval_with_evidence_text(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("cli_retrieval_evidence")
            tracker.record_document(
                doc_id="doc_evidence",
                source_path="/tmp/doc_evidence.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_evidence:chunk:0",
                document_id="doc_evidence",
                chunk_index=0,
                strategy="semantic",
                chunk_size=4,
                text_preview="base",
            )
            tracker.record_vector(
                vector_id="ev_v1",
                collection_name="support_kb",
                chunk_id="doc_evidence:chunk:0",
                embedding_model="m1",
            )
            tracker.complete_pipeline("success")

        result = cli(
            [
                "record-retrieval",
                "--db",
                self.db_path,
                "--collection",
                "support_kb",
                "--vector-id",
                "ev_v1",
                "--query-text",
                "q",
                "--final-answer",
                "a",
                "--evidence-text",
                "refunds allowed within 30 days",
            ]
        )
        self.assertEqual(result, 0)

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            trace = cli(
                [
                    "trace",
                    "--db",
                    self.db_path,
                    "--vector-id",
                    "ev_v1",
                    "--collection",
                    "support_kb",
                    "--format",
                    "json",
                    "--include-retrieval",
                ]
            )
        self.assertEqual(trace, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(
            payload["retrieval"]["metadata"]["evidence_text"], "refunds allowed within 30 days"
        )
        self.assertEqual(payload["evidence"]["chunk_text"], "refunds allowed within 30 days")

    def test_trace_support_assessment_detects_temporal_conflict(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("cli_support_conflict")
            tracker.record_document(
                doc_id="doc_support_conflict",
                source_path="/tmp/doc_support_conflict.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_support_conflict:chunk:0",
                document_id="doc_support_conflict",
                chunk_index=0,
                strategy="semantic",
                chunk_size=80,
                text_preview="Refunds are only allowed within 30 days for eligible defects.",
            )
            tracker.record_vector(
                vector_id="support_conflict_v1",
                collection_name="support_kb",
                chunk_id="doc_support_conflict:chunk:0",
                embedding_model="m1",
            )
            tracker.record_retrieval_event(
                query_id="support_q1",
                query_text="Can I get a refund after 90 days?",
                final_answer="Yes, refunds are allowed.",
                collection_name="support_kb",
                vector_id="support_conflict_v1",
                rank=1,
                score=0.88,
                metadata_json='{"evidence_text":"Refunds are only allowed within 30 days for eligible defects."}',
            )
            tracker.complete_pipeline("success")

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = cli(
                [
                    "trace",
                    "--db",
                    self.db_path,
                    "--vector-id",
                    "support_conflict_v1",
                    "--collection",
                    "support_kb",
                    "--format",
                    "json",
                    "--include-retrieval",
                ]
            )
        self.assertEqual(result, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["evidence"]["support_status"], "unsupported")
        self.assertIn("after 90 days", payload["evidence"]["support_reason"])

    def test_trace_support_assessment_polarity_only_is_unclear(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("cli_support_polarity")
            tracker.record_document(
                doc_id="doc_support_polarity",
                source_path="/tmp/doc_support_polarity.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_support_polarity:chunk:0",
                document_id="doc_support_polarity",
                chunk_index=0,
                strategy="semantic",
                chunk_size=60,
                text_preview="Refund policy allows refunds for defective products.",
            )
            tracker.record_vector(
                vector_id="support_polarity_v1",
                collection_name="support_kb",
                chunk_id="doc_support_polarity:chunk:0",
                embedding_model="m1",
            )
            tracker.record_retrieval_event(
                query_id="support_q2",
                query_text="Can I get a refund?",
                final_answer="Yes.",
                collection_name="support_kb",
                vector_id="support_polarity_v1",
                rank=1,
                score=0.55,
                metadata_json='{"evidence_text":"Refund policy allows refunds for defective products."}',
            )
            tracker.complete_pipeline("success")

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = cli(
                [
                    "trace",
                    "--db",
                    self.db_path,
                    "--vector-id",
                    "support_polarity_v1",
                    "--collection",
                    "support_kb",
                    "--format",
                    "json",
                    "--include-retrieval",
                ]
            )
        self.assertEqual(result, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["evidence"]["support_status"], "unclear")
        self.assertIn("polarity-only", payload["evidence"]["support_reason"])

    def test_trace_redact_preview_redacts_retrieval_evidence_text(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("cli_redact_evidence")
            tracker.record_document(
                doc_id="doc_redact_evidence",
                source_path="/tmp/private2.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_redact_evidence:chunk:0",
                document_id="doc_redact_evidence",
                chunk_index=0,
                strategy="semantic",
                chunk_size=20,
                text_preview="visible-preview-text",
            )
            tracker.record_vector(
                vector_id="r2",
                collection_name="support_kb",
                chunk_id="doc_redact_evidence:chunk:0",
                embedding_model="m3",
            )
            tracker.record_retrieval_event(
                query_id="qr_redact",
                query_text="q",
                final_answer="a",
                collection_name="support_kb",
                vector_id="r2",
                rank=1,
                score=0.8,
                metadata_json='{"evidence_text":"super-secret-evidence-text"}',
            )
            tracker.complete_pipeline("success")

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = cli(
                [
                    "trace",
                    "--vector-id",
                    "r2",
                    "--collection",
                    "support_kb",
                    "--db",
                    self.db_path,
                    "--redact-preview",
                    "--format",
                    "json",
                    "--include-retrieval",
                ]
            )
        self.assertEqual(result, 0)
        payload = json.loads(stdout.getvalue())
        self.assertNotIn("super-secret-evidence-text", stdout.getvalue())
        self.assertIn("[REDACTED:", payload["evidence"]["chunk_text"])
        self.assertIn(
            "[REDACTED:",
            payload["retrieval"]["metadata"]["evidence_text"],
        )

    def test_trace_redact_preview_does_not_change_support_assessment(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("cli_redact_preview_support_consistency")
            tracker.record_document(
                doc_id="doc_support_consistency",
                source_path="/tmp/doc_support_consistency.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_support_consistency:chunk:0",
                document_id="doc_support_consistency",
                chunk_index=0,
                strategy="semantic",
                chunk_size=80,
                text_preview="Refunds are only allowed within 30 days for eligible defects.",
            )
            tracker.record_vector(
                vector_id="support_consistency_v",
                collection_name="support_kb",
                chunk_id="doc_support_consistency:chunk:0",
                embedding_model="m1",
            )
            tracker.record_retrieval_event(
                query_id="support_consistency_q",
                query_text="Can I get a refund after 90 days?",
                final_answer="No, refunds are not allowed.",
                collection_name="support_kb",
                vector_id="support_consistency_v",
                rank=1,
                score=0.9,
            )
            tracker.complete_pipeline("success")

        plain_stdout = io.StringIO()
        with redirect_stdout(plain_stdout):
            plain_result = cli(
                [
                    "trace",
                    "--db",
                    self.db_path,
                    "--vector-id",
                    "support_consistency_v",
                    "--collection",
                    "support_kb",
                    "--format",
                    "json",
                    "--include-retrieval",
                ]
            )
        self.assertEqual(plain_result, 0)
        plain_payload = json.loads(plain_stdout.getvalue())

        redacted_stdout = io.StringIO()
        with redirect_stdout(redacted_stdout):
            redacted_result = cli(
                [
                    "trace",
                    "--db",
                    self.db_path,
                    "--vector-id",
                    "support_consistency_v",
                    "--collection",
                    "support_kb",
                    "--format",
                    "json",
                    "--include-retrieval",
                    "--redact-preview",
                ]
            )
        self.assertEqual(redacted_result, 0)
        redacted_payload = json.loads(redacted_stdout.getvalue())

        self.assertEqual(plain_payload["evidence"]["support_status"], "supported")
        self.assertEqual(
            redacted_payload["evidence"]["support_status"],
            plain_payload["evidence"]["support_status"],
        )

    def test_record_retrieval_evidence_text_rejects_non_object_metadata(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("cli_retrieval_evidence_bad")
            tracker.record_document(
                doc_id="doc_evidence_bad",
                source_path="/tmp/doc_evidence_bad.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_evidence_bad:chunk:0",
                document_id="doc_evidence_bad",
                chunk_index=0,
                strategy="semantic",
                chunk_size=4,
                text_preview="bad",
            )
            tracker.record_vector(
                vector_id="ev_bad_v1",
                collection_name="support_kb",
                chunk_id="doc_evidence_bad:chunk:0",
                embedding_model="m1",
            )
            tracker.complete_pipeline("success")

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = cli(
                [
                    "record-retrieval",
                    "--db",
                    self.db_path,
                    "--collection",
                    "support_kb",
                    "--vector-id",
                    "ev_bad_v1",
                    "--query-text",
                    "q",
                    "--metadata-json",
                    "[]",
                    "--evidence-text",
                    "text",
                ]
            )
        self.assertEqual(result, 2)
        self.assertIn("requires --metadata-json to be an object", stderr.getvalue())

    def test_record_qa_auto_records_from_question_answer(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("cli_record_qa")
            tracker.record_document(
                doc_id="doc_record_qa",
                source_path="/tmp/doc_record_qa.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_record_qa:chunk:refund",
                document_id="doc_record_qa",
                chunk_index=0,
                strategy="semantic",
                chunk_size=48,
                text_preview="Refunds are allowed only within 30 days for defects.",
            )
            tracker.record_chunk(
                chunk_id="doc_record_qa:chunk:warranty",
                document_id="doc_record_qa",
                chunk_index=1,
                strategy="semantic",
                chunk_size=36,
                text_preview="Warranty covers hardware defects only.",
            )
            tracker.record_vector(
                vector_id="qa_refund_v",
                collection_name="support_kb",
                chunk_id="doc_record_qa:chunk:refund",
                embedding_model="m1",
            )
            tracker.record_vector(
                vector_id="qa_warranty_v",
                collection_name="support_kb",
                chunk_id="doc_record_qa:chunk:warranty",
                embedding_model="m1",
            )
            tracker.complete_pipeline("success")

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = cli(
                [
                    "record-qa",
                    "--db",
                    self.db_path,
                    "--collection",
                    "support_kb",
                    "--question",
                    "Can I get a refund after 90 days?",
                    "--final-answer",
                    "Yes, refunds are allowed.",
                    "--top-k",
                    "1",
                ]
            )
        self.assertEqual(result, 0)
        self.assertIn("Recorded 1 retrieval event", stdout.getvalue())
        self.assertIn("vector_id=qa_refund_v", stdout.getvalue())

        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                """
                SELECT vector_id, rank, score, metadata_json
                FROM retrieval_events
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()
        finally:
            conn.close()

        self.assertIsNotNone(row)
        if row is None:
            return
        self.assertEqual(row[0], "qa_refund_v")
        self.assertEqual(row[1], 1)
        self.assertGreater(row[2], 0.0)
        metadata = json.loads(row[3])
        self.assertEqual(metadata["score_type"], "lexical_overlap_bootstrap")
        self.assertEqual(
            metadata["evidence_text"], "Refunds are allowed only within 30 days for defects."
        )

        report_stdout = io.StringIO()
        qa_report_path = str(Path(self.tmp.name) / "qa-auto.html")
        with redirect_stdout(report_stdout):
            report_result = cli(
                [
                    "report-qa",
                    "--db",
                    self.db_path,
                    "--question",
                    "Can I get a refund after 90 days?",
                    "--answer",
                    "Yes, refunds are allowed.",
                    "--collection",
                    "support_kb",
                    "--output",
                    qa_report_path,
                ]
            )
        self.assertEqual(report_result, 0)
        self.assertIn("Matched vector: qa_refund_v", report_stdout.getvalue())

    def test_record_qa_requires_positive_top_k(self) -> None:
        cli(["init", "--db", self.db_path])
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = cli(
                [
                    "record-qa",
                    "--db",
                    self.db_path,
                    "--collection",
                    "support_kb",
                    "--question",
                    "q",
                    "--final-answer",
                    "a",
                    "--top-k",
                    "0",
                ]
            )
        self.assertEqual(result, 2)
        self.assertIn("--top-k must be > 0", stderr.getvalue())

    def test_ask_trace_bootstraps_answer_and_writes_outputs(self) -> None:
        html_path = str(Path(self.tmp.name) / "ask-trace.html")
        json_path = str(Path(self.tmp.name) / "ask-trace.json")
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("cli_ask_trace")
            tracker.record_document(
                doc_id="doc_ask_trace",
                source_path="/tmp/doc_ask_trace.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_ask_trace:chunk:refund",
                document_id="doc_ask_trace",
                chunk_index=0,
                strategy="semantic",
                chunk_size=56,
                text_preview="Refunds are only allowed within 30 days for eligible defects.",
            )
            tracker.record_vector(
                vector_id="ask_refund_v",
                collection_name="support_kb",
                chunk_id="doc_ask_trace:chunk:refund",
                embedding_model="m1",
            )
            tracker.complete_pipeline("success")

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = cli(
                [
                    "ask-trace",
                    "--db",
                    self.db_path,
                    "--collection",
                    "support_kb",
                    "--question",
                    "Can I get a refund after 90 days?",
                    "--top-k",
                    "1",
                    "--output",
                    html_path,
                    "--json-output",
                    json_path,
                ]
            )

        self.assertEqual(result, 0)
        self.assertTrue(Path(html_path).exists())
        self.assertTrue(Path(json_path).exists())
        self.assertIn("Final answer was bootstrapped", stdout.getvalue())
        self.assertIn("Matched vector: ask_refund_v", stdout.getvalue())

        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        self.assertEqual(payload["match"]["trace"]["vector"]["id"], "ask_refund_v")
        self.assertEqual(payload["match"]["retrieval"]["trace_mode"], "bootstrap")
        self.assertEqual(
            payload["match"]["retrieval"]["final_answer"],
            "Refunds are only allowed within 30 days for eligible defects.",
        )
        self.assertEqual(
            payload["match"]["evidence"]["chunk_text"],
            "Refunds are only allowed within 30 days for eligible defects.",
        )

    def test_ask_trace_with_provided_answer_uses_it(self) -> None:
        json_path = str(Path(self.tmp.name) / "ask-trace-provided.json")
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("cli_ask_trace_provided")
            tracker.record_document(
                doc_id="doc_ask_trace_provided",
                source_path="/tmp/doc_ask_trace_provided.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_ask_trace_provided:chunk:0",
                document_id="doc_ask_trace_provided",
                chunk_index=0,
                strategy="semantic",
                chunk_size=40,
                text_preview="Refunds are limited to 30 days.",
            )
            tracker.record_vector(
                vector_id="ask_provided_v",
                collection_name="support_kb",
                chunk_id="doc_ask_trace_provided:chunk:0",
                embedding_model="m1",
            )
            tracker.complete_pipeline("success")

        result = cli(
            [
                "ask-trace",
                "--db",
                self.db_path,
                "--collection",
                "support_kb",
                "--question",
                "Can I get a refund after 90 days?",
                "--final-answer",
                "Yes, refunds are allowed.",
                "--top-k",
                "1",
                "--output",
                self.report_path,
                "--json-output",
                json_path,
            ]
        )
        self.assertEqual(result, 0)
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        self.assertEqual(payload["match"]["retrieval"]["final_answer"], "Yes, refunds are allowed.")

    def test_ask_trace_rejects_invalid_match_index(self) -> None:
        cli(["init", "--db", self.db_path])
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = cli(
                [
                    "ask-trace",
                    "--db",
                    self.db_path,
                    "--collection",
                    "support_kb",
                    "--question",
                    "q",
                    "--match-index",
                    "0",
                ]
            )
        self.assertEqual(result, 2)
        self.assertIn("--match-index must be > 0", stderr.getvalue())

    def test_report_include_retrieval_renders_context(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("cli_report_retrieval")
            tracker.record_document(
                doc_id="doc_report_retrieval",
                source_path="/tmp/doc_report_retrieval.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_report_retrieval:chunk:0",
                document_id="doc_report_retrieval",
                chunk_index=0,
                strategy="semantic",
                chunk_size=4,
                text_preview="text",
            )
            tracker.record_vector(
                vector_id="rep_ret_v",
                collection_name="support_kb",
                chunk_id="doc_report_retrieval:chunk:0",
                embedding_model="m1",
            )
            tracker.record_retrieval_event(
                query_id="q1",
                query_text="where is refund policy",
                final_answer="look at doc",
                collection_name="support_kb",
                vector_id="rep_ret_v",
                rank=1,
                score=0.42,
                metadata_json='{"request_id":"r1"}',
            )
            tracker.complete_pipeline("success")

        result = cli(
            [
                "report",
                "--db",
                self.db_path,
                "--vector-id",
                "rep_ret_v",
                "--collection",
                "support_kb",
                "--output",
                self.report_path,
                "--include-retrieval",
            ]
        )
        self.assertEqual(result, 0)
        html = Path(self.report_path).read_text(encoding="utf-8")
        self.assertIn("Retrieval Context", html)
        self.assertIn("where is refund policy", html)

    def test_trace_qa_json_returns_match(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("cli_trace_qa")
            tracker.record_document(
                doc_id="doc_trace_qa",
                source_path="/tmp/doc_trace_qa.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_trace_qa:chunk:0",
                document_id="doc_trace_qa",
                chunk_index=0,
                strategy="semantic",
                chunk_size=8,
                text_preview="trace qa",
            )
            tracker.record_vector(
                vector_id="trace_qa_v1",
                collection_name="support_kb",
                chunk_id="doc_trace_qa:chunk:0",
                embedding_model="m1",
            )
            tracker.record_retrieval_event(
                query_id="qa_1",
                query_text="Where is refund policy?",
                final_answer="It is in support docs.",
                collection_name="support_kb",
                vector_id="trace_qa_v1",
                rank=1,
                score=0.91,
            )
            tracker.complete_pipeline("success")

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = cli(
                [
                    "trace-qa",
                    "--db",
                    self.db_path,
                    "--question",
                    "Where is refund policy?",
                    "--answer",
                    "It is in support docs.",
                    "--collection",
                    "support_kb",
                    "--format",
                    "json",
                ]
            )
        self.assertEqual(result, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(len(payload["matches"]), 1)
        self.assertEqual(payload["matches"][0]["trace"]["vector"]["id"], "trace_qa_v1")
        self.assertEqual(payload["matches"][0]["evidence"]["chunk_text"], "trace qa")

    def test_trace_qa_returns_not_found(self) -> None:
        cli(["init", "--db", self.db_path])
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = cli(
                [
                    "trace-qa",
                    "--db",
                    self.db_path,
                    "--question",
                    "missing question",
                ]
            )
        self.assertEqual(result, 1)
        self.assertIn("No retrieval events matched", stderr.getvalue())

    def test_report_qa_generates_html(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("cli_report_qa")
            tracker.record_document(
                doc_id="doc_report_qa",
                source_path="/tmp/doc_report_qa.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_report_qa:chunk:0",
                document_id="doc_report_qa",
                chunk_index=0,
                strategy="semantic",
                chunk_size=11,
                text_preview="report qa txt",
            )
            tracker.record_vector(
                vector_id="report_qa_v1",
                collection_name="support_kb",
                chunk_id="doc_report_qa:chunk:0",
                embedding_model="m1",
            )
            tracker.record_retrieval_event(
                query_id="rq_1",
                query_text="q report",
                final_answer="a report",
                collection_name="support_kb",
                vector_id="report_qa_v1",
                rank=1,
                score=0.77,
            )
            tracker.complete_pipeline("success")

        result = cli(
            [
                "report-qa",
                "--db",
                self.db_path,
                "--question",
                "q report",
                "--answer",
                "a report",
                "--collection",
                "support_kb",
                "--output",
                self.report_path,
            ]
        )
        self.assertEqual(result, 0)
        html = Path(self.report_path).read_text(encoding="utf-8")
        self.assertIn("Answer Evidence", html)
        self.assertIn("report_qa_v1", html)

    def test_report_qa_generates_html_and_json(self) -> None:
        json_path = str(Path(self.tmp.name) / "qa-trace.json")
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("cli_report_qa_json")
            tracker.record_document(
                doc_id="doc_report_qa_json",
                source_path="/tmp/doc_report_qa_json.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_report_qa_json:chunk:0",
                document_id="doc_report_qa_json",
                chunk_index=0,
                strategy="semantic",
                chunk_size=12,
                text_preview="qa json text",
            )
            tracker.record_vector(
                vector_id="report_qa_json_v1",
                collection_name="support_kb",
                chunk_id="doc_report_qa_json:chunk:0",
                embedding_model="m1",
            )
            tracker.record_retrieval_event(
                query_id="rq_json_1",
                query_text="q report json",
                final_answer="a report json",
                collection_name="support_kb",
                vector_id="report_qa_json_v1",
                rank=1,
                score=0.88,
            )
            tracker.complete_pipeline("success")

        result = cli(
            [
                "report-qa",
                "--db",
                self.db_path,
                "--question",
                "q report json",
                "--answer",
                "a report json",
                "--collection",
                "support_kb",
                "--output",
                self.report_path,
                "--json-output",
                json_path,
            ]
        )
        self.assertEqual(result, 0)
        self.assertTrue(Path(self.report_path).exists())
        self.assertTrue(Path(json_path).exists())

        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["match"]["trace"]["vector"]["id"], "report_qa_json_v1")
        self.assertEqual(payload["match"]["evidence"]["chunk_text"], "qa json text")

    def test_report_qa_returns_not_found(self) -> None:
        cli(["init", "--db", self.db_path])
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = cli(
                [
                    "report-qa",
                    "--db",
                    self.db_path,
                    "--question",
                    "unknown",
                    "--output",
                    self.report_path,
                ]
            )
        self.assertEqual(result, 1)
        self.assertIn("No retrieval events matched", stderr.getvalue())

    def test_trace_plain_output(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("cli_plain")
            tracker.record_document(
                doc_id="doc_plain",
                source_path="/tmp/doc.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_plain:chunk:0",
                document_id="doc_plain",
                chunk_index=0,
                strategy="fixed-size",
                chunk_size=10,
                text_preview="plain-text",
            )
            tracker.record_vector(
                vector_id="p1",
                collection_name="support_kb",
                chunk_id="doc_plain:chunk:0",
                embedding_model="m2",
            )
            tracker.complete_pipeline("success")

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = cli(
                [
                    "trace",
                    "--vector-id",
                    "p1",
                    "--collection",
                    "support_kb",
                    "--db",
                    self.db_path,
                    "--plain",
                ]
            )
        self.assertEqual(result, 0)
        output = stdout.getvalue()
        self.assertIn("vector.id=p1", output)
        self.assertIn("chunk.preview=plain-text", output)

    def test_trace_redact_preview(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("cli_redact")
            tracker.record_document(
                doc_id="doc_redact",
                source_path="/tmp/private.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_redact:chunk:0",
                document_id="doc_redact",
                chunk_index=0,
                strategy="semantic",
                chunk_size=17,
                text_preview="sensitive-preview",
            )
            tracker.record_vector(
                vector_id="r1",
                collection_name="support_kb",
                chunk_id="doc_redact:chunk:0",
                embedding_model="m3",
            )
            tracker.complete_pipeline("success")

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = cli(
                [
                    "trace",
                    "--vector-id",
                    "r1",
                    "--collection",
                    "support_kb",
                    "--db",
                    self.db_path,
                    "--redact-preview",
                ]
            )
        self.assertEqual(result, 0)
        output = stdout.getvalue()
        self.assertIn("[REDACTED:", output)
        self.assertNotIn("sensitive-preview", output)

    def test_connect_surfaces_connection_errors(self) -> None:
        stderr = io.StringIO()
        with patch("connectors.qdrant.test_connection", side_effect=RuntimeError("boom")):
            with redirect_stderr(stderr):
                result = cli(
                    [
                        "connect",
                        "--qdrant-url",
                        "http://localhost:6333",
                        "--qdrant-collection",
                        "support_kb",
                    ]
                )
        self.assertEqual(result, 2)
        self.assertIn("boom", stderr.getvalue())

    def test_trace_rejects_plain_with_json_format(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = cli(
                [
                    "trace",
                    "--vector-id",
                    "x",
                    "--db",
                    self.db_path,
                    "--format",
                    "json",
                    "--plain",
                ]
            )
        self.assertEqual(result, 2)
        self.assertIn("cannot be used", stderr.getvalue())

    def test_report_redact_preview(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("cli_report_redact")
            tracker.record_document(
                doc_id="doc_report_redact",
                source_path="/tmp/private.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_report_redact:chunk:0",
                document_id="doc_report_redact",
                chunk_index=0,
                strategy="semantic",
                chunk_size=16,
                text_preview="super-secret-text",
            )
            tracker.record_vector(
                vector_id="rr1",
                collection_name="support_kb",
                chunk_id="doc_report_redact:chunk:0",
                embedding_model="m4",
            )
            tracker.complete_pipeline("success")

        result = cli(
            [
                "report",
                "--vector-id",
                "rr1",
                "--collection",
                "support_kb",
                "--db",
                self.db_path,
                "--output",
                self.report_path,
                "--redact-preview",
            ]
        )
        self.assertEqual(result, 0)
        rendered = Path(self.report_path).read_text(encoding="utf-8")
        self.assertIn("[REDACTED:", rendered)
        self.assertNotIn("super-secret-text", rendered)

    def test_report_redact_preview_redacts_retrieval_evidence_text(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("cli_report_redact_evidence")
            tracker.record_document(
                doc_id="doc_report_redact_evidence",
                source_path="/tmp/private3.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_report_redact_evidence:chunk:0",
                document_id="doc_report_redact_evidence",
                chunk_index=0,
                strategy="semantic",
                chunk_size=16,
                text_preview="public-preview",
            )
            tracker.record_vector(
                vector_id="rr2",
                collection_name="support_kb",
                chunk_id="doc_report_redact_evidence:chunk:0",
                embedding_model="m4",
            )
            tracker.record_retrieval_event(
                query_id="qr_report_redact",
                query_text="q",
                final_answer="a",
                collection_name="support_kb",
                vector_id="rr2",
                rank=1,
                score=0.9,
                metadata_json='{"evidence_text":"ultra-secret-report-evidence"}',
            )
            tracker.complete_pipeline("success")

        result = cli(
            [
                "report",
                "--vector-id",
                "rr2",
                "--collection",
                "support_kb",
                "--db",
                self.db_path,
                "--output",
                self.report_path,
                "--redact-preview",
                "--include-retrieval",
            ]
        )
        self.assertEqual(result, 0)
        rendered = Path(self.report_path).read_text(encoding="utf-8")
        self.assertNotIn("ultra-secret-report-evidence", rendered)
        self.assertIn("[REDACTED:", rendered)

    def test_trace_redact_retrieval_redacts_query_and_answer(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("cli_redact_retrieval")
            tracker.record_document(
                doc_id="doc_redact_retrieval",
                source_path="/tmp/doc_redact_retrieval.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_redact_retrieval:chunk:0",
                document_id="doc_redact_retrieval",
                chunk_index=0,
                strategy="semantic",
                chunk_size=20,
                text_preview="chunk text",
            )
            tracker.record_vector(
                vector_id="rr_redact_v",
                collection_name="support_kb",
                chunk_id="doc_redact_retrieval:chunk:0",
                embedding_model="m1",
            )
            tracker.record_retrieval_event(
                query_id="qr_redact_fields",
                query_text="secret query text",
                final_answer="secret answer text",
                collection_name="support_kb",
                vector_id="rr_redact_v",
                rank=1,
                score=0.6,
                metadata_json='{"query_text":"secret query text","answer":"secret answer text"}',
            )
            tracker.complete_pipeline("success")

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = cli(
                [
                    "trace",
                    "--db",
                    self.db_path,
                    "--vector-id",
                    "rr_redact_v",
                    "--collection",
                    "support_kb",
                    "--format",
                    "json",
                    "--include-retrieval",
                    "--redact-retrieval",
                ]
            )
        self.assertEqual(result, 0)
        payload = json.loads(stdout.getvalue())
        self.assertIn("[REDACTED:", payload["retrieval"]["query_text"])
        self.assertIn("[REDACTED:", payload["retrieval"]["final_answer"])
        self.assertIn("[REDACTED:", payload["retrieval"]["metadata"]["query_text"])
        self.assertIn("[REDACTED:", payload["retrieval"]["metadata"]["answer"])
        self.assertNotIn("secret query text", stdout.getvalue())
        self.assertNotIn("secret answer text", stdout.getvalue())

    def test_trace_redact_retrieval_redacts_non_object_metadata(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("cli_redact_retrieval_list_metadata")
            tracker.record_document(
                doc_id="doc_redact_list_metadata",
                source_path="/tmp/doc_redact_list_metadata.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_redact_list_metadata:chunk:0",
                document_id="doc_redact_list_metadata",
                chunk_index=0,
                strategy="semantic",
                chunk_size=16,
                text_preview="chunk list text",
            )
            tracker.record_vector(
                vector_id="rr_redact_list_v",
                collection_name="support_kb",
                chunk_id="doc_redact_list_metadata:chunk:0",
                embedding_model="m1",
            )
            tracker.record_retrieval_event(
                query_id="qr_redact_list",
                query_text="secret question list",
                final_answer="secret answer list",
                collection_name="support_kb",
                vector_id="rr_redact_list_v",
                rank=1,
                score=0.6,
                metadata_json='["secret-list-item"]',
            )
            tracker.complete_pipeline("success")

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = cli(
                [
                    "trace",
                    "--db",
                    self.db_path,
                    "--vector-id",
                    "rr_redact_list_v",
                    "--collection",
                    "support_kb",
                    "--format",
                    "json",
                    "--include-retrieval",
                    "--redact-retrieval",
                ]
            )
        self.assertEqual(result, 0)
        payload = json.loads(stdout.getvalue())
        self.assertIn("[REDACTED:", payload["retrieval"]["metadata"])
        self.assertNotIn("secret-list-item", stdout.getvalue())

    def test_report_redact_retrieval_redacts_query_and_answer_in_html(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("cli_report_redact_retrieval")
            tracker.record_document(
                doc_id="doc_report_redact_retrieval",
                source_path="/tmp/doc_report_redact_retrieval.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_report_redact_retrieval:chunk:0",
                document_id="doc_report_redact_retrieval",
                chunk_index=0,
                strategy="semantic",
                chunk_size=20,
                text_preview="chunk text",
            )
            tracker.record_vector(
                vector_id="rr_report_fields_v",
                collection_name="support_kb",
                chunk_id="doc_report_redact_retrieval:chunk:0",
                embedding_model="m1",
            )
            tracker.record_retrieval_event(
                query_id="qr_report_redact_fields",
                query_text="very secret question",
                final_answer="very secret answer",
                collection_name="support_kb",
                vector_id="rr_report_fields_v",
                rank=1,
                score=0.91,
            )
            tracker.complete_pipeline("success")

        result = cli(
            [
                "report",
                "--db",
                self.db_path,
                "--vector-id",
                "rr_report_fields_v",
                "--collection",
                "support_kb",
                "--output",
                self.report_path,
                "--include-retrieval",
                "--redact-retrieval",
            ]
        )
        self.assertEqual(result, 0)
        rendered = Path(self.report_path).read_text(encoding="utf-8")
        self.assertIn("[REDACTED:", rendered)
        self.assertNotIn("very secret question", rendered)
        self.assertNotIn("very secret answer", rendered)

    def test_trace_redact_retrieval_redacts_assessment_text(self) -> None:
        with LineageTracker(self.db_path, autoinit=True) as tracker:
            tracker.start_pipeline("cli_redact_assessment")
            tracker.record_document(
                doc_id="doc_redact_assessment",
                source_path="/tmp/doc_redact_assessment.txt",
                source_type="local",
                version="v1",
            )
            tracker.record_chunk(
                chunk_id="doc_redact_assessment:chunk:0",
                document_id="doc_redact_assessment",
                chunk_index=0,
                strategy="semantic",
                chunk_size=80,
                text_preview="Refunds are only allowed within 30 days for eligible defects.",
            )
            tracker.record_vector(
                vector_id="rr_assessment_v",
                collection_name="support_kb",
                chunk_id="doc_redact_assessment:chunk:0",
                embedding_model="m1",
            )
            tracker.record_retrieval_event(
                query_id="qr_assessment",
                query_text="Can I get a refund after 90 days?",
                final_answer="Yes, refunds are allowed.",
                collection_name="support_kb",
                vector_id="rr_assessment_v",
                rank=1,
                score=0.9,
                metadata_json='{"evidence_text":"Refunds are only allowed within 30 days for eligible defects."}',
            )
            tracker.complete_pipeline("success")

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = cli(
                [
                    "trace",
                    "--db",
                    self.db_path,
                    "--vector-id",
                    "rr_assessment_v",
                    "--collection",
                    "support_kb",
                    "--format",
                    "json",
                    "--include-retrieval",
                    "--redact-retrieval",
                ]
            )
        self.assertEqual(result, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["evidence"]["support_status"], "unsupported")
        self.assertEqual(
            payload["evidence"]["support_reason"],
            "Assessment computed; retrieval context redacted.",
        )
        self.assertTrue(payload["evidence"]["support_details"]["redacted"])
        self.assertNotIn("after 90 days", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
