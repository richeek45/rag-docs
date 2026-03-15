-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. The Papers table (Metadata)
CREATE TABLE IF NOT EXISTS papers (
    id SERIAL PRIMARY KEY,
    title TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE paper_chunks (
    id SERIAL PRIMARY KEY,
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER, 
    embedding vector(4096) -- Qwen3-Embedding-8B uses 4096 dimensions
);

CREATE INDEX ON papers(title);
CREATE INDEX ON paper_chunks USING hnsw (embedding vector_cosine_ops);