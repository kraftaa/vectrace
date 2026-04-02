PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    source_type TEXT NOT NULL,
    version TEXT,
    content_hash TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    pipeline_run_id TEXT,
    chunk_index INTEGER NOT NULL,
    strategy TEXT NOT NULL,
    chunk_size INTEGER,
    text_preview TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY (pipeline_run_id) REFERENCES pipeline_runs(id)
);

CREATE TABLE IF NOT EXISTS vectors (
    id TEXT NOT NULL,
    collection_name TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    pipeline_run_id TEXT,
    embedding_model TEXT NOT NULL,
    model_version TEXT,
    batch_id TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (collection_name, id),
    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
    FOREIGN KEY (pipeline_run_id) REFERENCES pipeline_runs(id)
);

CREATE TABLE IF NOT EXISTS retrieval_events (
    id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    final_answer TEXT,
    collection_name TEXT NOT NULL,
    vector_id TEXT NOT NULL,
    rank INTEGER,
    score REAL,
    metadata_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (collection_name, vector_id) REFERENCES vectors(collection_name, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_pipeline_run_id ON chunks(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_vectors_id ON vectors(id);
CREATE INDEX IF NOT EXISTS idx_vectors_chunk_id ON vectors(chunk_id);
CREATE INDEX IF NOT EXISTS idx_vectors_pipeline_run_id ON vectors(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_events_vector ON retrieval_events(collection_name, vector_id, created_at);
CREATE INDEX IF NOT EXISTS idx_retrieval_events_query ON retrieval_events(query_id, rank);
