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


if __name__ == "__main__":
    unittest.main()
