-- name: CreatePaper :one
INSERT INTO papers (title)
VALUES ($1)
RETURNING id, title, created_at;

-- name: CreatePaperChunk :one
INSERT INTO paper_chunks (paper_id, content, chunk_index, embedding)
VALUES ($1, $2, $3, $4)
RETURNING *;

-- name: GetPaperByID :one
SELECT * FROM papers
WHERE id = $1 LIMIT 1;

-- name: ListChunksByPaper :many
SELECT * FROM paper_chunks
WHERE paper_id = $1
ORDER BY chunk_index ASC;

-- name: SearchSimilarChunks :many
-- We use the <=> operator for cosine distance (smaller is better)
SELECT 
    pc.id, 
    pc.content, 
    pc.paper_id,
    p.title as paper_title,
    1 - (pc.embedding <=> $1) AS similarity
FROM paper_chunks pc
JOIN papers p ON pc.paper_id = p.id
ORDER BY pc.embedding <=> $1
LIMIT $2;

-- name: DeletePaper :exec
DELETE FROM papers
WHERE id = $1;