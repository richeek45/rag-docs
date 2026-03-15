package main

import (
	"bufio"
	"bytes"
	"context"
	_ "embed"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/blevesearch/segment"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/joho/godotenv"
	"github.com/parakeet-nest/parakeet/completion"
	"github.com/parakeet-nest/parakeet/embeddings"
	"github.com/parakeet-nest/parakeet/enums/option"
	"github.com/parakeet-nest/parakeet/llm"
	"github.com/pgvector/pgvector-go"
	"github.com/richeek45/rag-docs/queries/db"
)

type Chunker struct {
	MaxChunkChars int
	OverlapCount  int
}

func normalize(v []float32) []float32 {
	var sum float64
	for _, val := range v {
		sum += float64(val * val)
	}
	magnitude := math.Sqrt(sum)
	if magnitude == 0 {
		return v
	}

	res := make([]float32, len(v))
	for i, val := range v {
		res[i] = float32(float64(val) / magnitude)
	}
	return res
}

func ConvertMarkdownToPDF(filePath string, outputFile string) error {
	cmd := exec.Command("pdftotext", "-layout", filePath, outputFile)

	err := cmd.Run()
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err != nil {
		fmt.Printf("Error during conversion: %v\n", err)
		fmt.Printf("Error: %s\n", stderr.String())
		return err
	}

	fmt.Printf("Successfully converted %s to %s\n", filePath, outputFile)
	return nil
}

func DocumentVectorChunking(title string, query *db.Queries) error {
	file, err := os.Open(title)
	if err != nil {
		log.Fatal(err)
		return err
	}

	defer file.Close()

	chunker := Chunker{MaxChunkChars: 1200, OverlapCount: 2}
	err = chunker.ProcessAndSave(context.Background(), query, title, file)
	if err != nil {
		log.Fatalln("😡:", err)
		return err
	}
	return nil
}

func (c *Chunker) ProcessAndSave(ctx context.Context, q *db.Queries, title string, r io.Reader) error {
	// paper, err := q.CreatePaper(ctx, title)
	// if err != nil {
	// 	return err
	// }
	paper, err := q.GetPaperByID(ctx, 1)
	if err != nil {
		return err
	}

	scanner := bufio.NewScanner(r)
	var currentBuffer []string
	var currentLen int
	chunkIndex := 0
	currentHeader := ""

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		if strings.HasPrefix(line, "#") {
			if len(currentBuffer) > 0 {
				if err := c.flush(ctx, q, paper.ID, &currentBuffer, &currentLen, &chunkIndex, currentHeader); err != nil {
					return err
				}
			}
			currentHeader = line
			continue
		}

		seg := segment.NewSegmenterDirect([]byte(line))
		for seg.Segment() {
			if seg.Type() != segment.Letter {
				continue
			}

			sentence := strings.TrimSpace(seg.Text())
			currentBuffer = append(currentBuffer, sentence)
			currentLen += len(sentence)

		}

		if currentLen > c.MaxChunkChars {
			if err := c.flush(ctx, q, paper.ID, &currentBuffer, &currentLen, &chunkIndex, currentHeader); err != nil {
				return err
			}
		}
	}

	if len(currentBuffer) > 0 {
		return c.flush(ctx, q, paper.ID, &currentBuffer, &currentLen, &chunkIndex, currentHeader)
	}

	return scanner.Err()
}

func (c *Chunker) flush(ctx context.Context, q *db.Queries, paperId int32, buffer *[]string, currentLen *int, index *int, header string) error {
	ollamaUrl := "http://localhost:11434"
	embeddingsModel := "qwen3-embedding:8b" // This model is for the embeddings of the documents

	chunkText := strings.Join(*buffer, " ")

	if header != "" {
		chunkText = header + "\n" + chunkText
	}

	embedding, err := embeddings.CreateEmbedding(
		ollamaUrl,
		llm.Query4Embedding{
			Model:  embeddingsModel,
			Prompt: chunkText,
		},
		strconv.Itoa(*index),
	)
	if err != nil {
		fmt.Println("😡:", err)
	}

	targetDim := 2000
	var finalEmbedding []float32
	if len(embedding.Embedding) > targetDim {
		truncated := embedding.Embedding[:targetDim]

		vec32 := make([]float32, len(truncated))
		for i, vec := range truncated {
			vec32[i] = float32(vec)
		}

		// 2. Re-normalize to maintain search accuracy
		finalEmbedding = normalize(vec32)

		// 3. Save finalEmbedding to DB
		// Ensure your SQL table matches: embedding vector(1536)
	}

	fmt.Println("Creating chunk")
	_, err = q.CreatePaperChunk(ctx, db.CreatePaperChunkParams{
		Content:    chunkText,
		ChunkIndex: pgtype.Int4{Int32: int32(*index), Valid: true},
		Embedding:  pgvector.NewVector(finalEmbedding),
		PaperID:    pgtype.Int4{Int32: paperId, Valid: true},
	})
	if err != nil {
		log.Println(err)
		return err
	}

	overlapStart := len(*buffer) - c.OverlapCount
	if overlapStart < 0 {
		overlapStart = 0
	}

	remaining := (*buffer)[overlapStart:]
	*buffer = make([]string, len(remaining))
	copy(*buffer, remaining)
	*currentLen = 0
	for _, s := range *buffer {
		*currentLen += len(s)
	}
	*index++
	return nil
}

func Config() *pgxpool.Config {
	err := godotenv.Load()
	if err != nil {
		log.Println("Error loading .env file, assuming env vars set manually")
	}
	const defaultMaxConns = int32(4)
	const defaultMinConns = int32(0)
	const defaultMaxConnLifetime = time.Hour
	const defaultMaxConnIdleTime = time.Minute * 30
	const defaultHealthCheckPeriod = time.Minute
	const defaultConnectTimeout = time.Second * 5

	dbHost := os.Getenv("DB_HOST")
	dbName := os.Getenv("DB_NAME")
	dbUser := os.Getenv("DB_USER")
	dbPassword := os.Getenv("DB_PASSWORD")
	dbPort := os.Getenv("DB_PORT")
	sslMode := os.Getenv("SSL_MODE")

	fmt.Println(dbHost, dbName, dbUser, dbPassword, dbPort, sslMode)

	dsn := fmt.Sprintf("host=%s user=%s password=%s dbname=%s port=%s sslmode=%s", dbHost, dbUser, dbPassword, dbName, dbPort, sslMode)

	dbConfig, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		log.Fatal("Failed to create a config, error: ", err)
	}

	dbConfig.MaxConns = defaultMaxConns
	dbConfig.MinConns = defaultMinConns
	dbConfig.MaxConnLifetime = defaultMaxConnLifetime
	dbConfig.MaxConnIdleTime = defaultMaxConnIdleTime
	dbConfig.HealthCheckPeriod = defaultHealthCheckPeriod
	dbConfig.ConnConfig.ConnectTimeout = defaultConnectTimeout

	dbConfig.PrepareConn = func(ctx context.Context, c *pgx.Conn) (bool, error) {
		// log.Println("Before acquiring the connection pool to the database!!")
		return true, nil
	}

	dbConfig.AfterRelease = func(c *pgx.Conn) bool {
		// log.Println("After releasing the connection pool to the database!!")
		return true
	}

	dbConfig.BeforeClose = func(c *pgx.Conn) {
		log.Println("Closed the connection pool to the database!!")
	}
	return dbConfig
}

func main() {
	connPool, err := pgxpool.NewWithConfig(context.Background(), Config())
	if err != nil {
		log.Fatalf("Unable to connect to database: %v\n", err)
	}
	connection, err := connPool.Acquire(context.Background())
	if err != nil {
		log.Fatal("Error while acquiring connection from the database pool!!")
	}
	defer connection.Release()

	err = connection.Ping(context.Background())
	if err != nil {
		log.Fatal("Could not ping database")
	}

	fmt.Println("Connected to the database!!")

	query := db.New(connPool)
	defer connPool.Close()

	filePath := os.Getenv("FILE_PATH")
	parts := strings.Split(filePath, "/")
	title := parts[len(parts)-1]
	paperRow, _ := query.GetPaperByTitle(context.Background(), title)
	fmt.Println(paperRow.Title != title)
	if paperRow.Title != title {
		err = ConvertMarkdownToPDF(filePath, title)
		if err == nil {
			err = DocumentVectorChunking(title, query)
		}
		if err != nil {
			log.Fatal(err)
			return
		}
	}

	ollamaUrl := "http://localhost:11434"
	embeddingsModel := "qwen3-embedding:8b"
	chatModel := "qwen3:8b" //"llama3.2:latest"

	systemContent := `### ROLE
		You are an expert Research Assistant. Your goal is to provide accurate, concise, and context-aware answers based strictly on the provided document excerpts. You specialize in synthesizing information from diverse subjects, ranging from technical engineering to economics and law.

		### TASK
		1.  **Analyze the Context:** Review the provided snippets retrieved from the PDF database.
		2.  **Synthesize Knowledge:** If a question spans multiple documents or subjects, connect the concepts logically.
		3.  **Identify Constraints:** * If the answer is explicitly in the context, provide it with high detail.
			* If the context is insufficient, state: "The provided documents do not contain enough information to answer this specifically," then offer a brief answer based on general knowledge *only* if requested.
			* If the documents provide conflicting information, highlight the discrepancy (e.g., "Source A states X, while Source B suggests Y").

		### RESPONSE GUIDELINES
		* **Structure:** Use Markdown (bolding, bullet points, and headers) to make complex technical data readable.
		* **Tone:** Professional, objective, and analytical.
		* **Accuracy:** Do not hallucinate facts or dates not present in the text.
		* **Citations:** When possible, refer to the specific PDF or section name provided in the context metadata.

		### HANDLING TECHNICAL CONTENT
		* **Equations:** Use LaTeX for mathematical formulas.
		* **Code:** If the context contains code or logic, ensure indentation is preserved.
		* **Terminology:** Maintain the specific jargon used in the source material (e.g., if a document uses "idempotency" or "liquidity preference," use those exact terms in the explanation).

		---
		### USER QUERY:
		{user_query}

		### RETRIEVED CONTEXT:
		{context_chunks}`

	options := llm.SetOptions(map[string]any{
		option.Temperature: 0.0,
	})

	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Println("\n Ask a question or type exit?")
		if !scanner.Scan() {
			break
		}

		userContent := scanner.Text()
		if userContent == "exit" {
			os.Exit(1)
			break
		}

		embeddingFromQuestion, err := embeddings.CreateEmbedding(
			ollamaUrl,
			llm.Query4Embedding{
				Model:  embeddingsModel,
				Prompt: userContent,
			},
			"question",
		)
		if err != nil {
			log.Fatalln(err)
		}

		targetDim := 2000
		var finalEmbedding []float32
		if len(embeddingFromQuestion.Embedding) > targetDim {
			truncated := embeddingFromQuestion.Embedding[:targetDim]

			vec32 := make([]float32, len(truncated))
			for i, vec := range truncated {
				vec32[i] = float32(vec)
			}

			finalEmbedding = normalize(vec32)
		}

		similarityRows, _ := query.SearchSimilarChunks(
			context.Background(),
			db.SearchSimilarChunksParams{
				Embedding: pgvector.NewVector(finalEmbedding),
				Limit:     5,
			})

		var contextBuilder strings.Builder
		for _, row := range similarityRows {
			fmt.Fprintf(&contextBuilder, "\n--- Source: %s ---\n%s\n", row.PaperTitle, row.Content)
		}
		documentsContent := contextBuilder.String()

		llmQuery := llm.Query{
			Model: chatModel,
			Messages: []llm.Message{
				{Role: "system", Content: systemContent},
				{Role: "system", Content: documentsContent},
				{Role: "user", Content: userContent},
			},
			Options: options,
		}

		_, err = completion.ChatStream(ollamaUrl, llmQuery,
			func(answer llm.Answer) error {
				fmt.Print(answer.Message.Content)
				return nil
			})

		if err != nil {
			log.Fatal(err)
		}

	}

}
