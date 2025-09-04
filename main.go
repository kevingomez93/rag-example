package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/sashabaranov/go-openai"
)

// OpenAIClientImpl implements the OpenAIClient interface
type OpenAIClientImpl struct {
	client *openai.Client
}

func (o *OpenAIClientImpl) ChatCompletion(model string, messages []Message) (string, error) {
	var openaiMessages []openai.ChatCompletionMessage
	for _, msg := range messages {
		openaiMessages = append(openaiMessages, openai.ChatCompletionMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	resp, err := o.client.CreateChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model:    model,
			Messages: openaiMessages,
		},
	)
	if err != nil {
		return "", err
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no response from OpenAI")
	}

	return resp.Choices[0].Message.Content, nil
}

// MilvusClientImpl implements the MilvusClient interface
type MilvusClientImpl struct {
	client         client.Client
	collectionName string
}

func (m *MilvusClientImpl) InsertDocuments(texts, sources []string) bool {
	ctx := context.Background()

	// Check if collection exists, create if not
	hasCollection, err := m.client.HasCollection(ctx, m.collectionName)
	if err != nil {
		log.Printf("Error checking collection: %v", err)
		return false
	}

	if !hasCollection {
		// Create collection schema
		schema := &entity.Schema{
			CollectionName: m.collectionName,
			Description:    "RAG documents collection",
			Fields: []*entity.Field{
				{
					Name:       "id",
					DataType:   entity.FieldTypeInt64,
					PrimaryKey: true,
					AutoID:     true,
				},
				{
					Name:     "text",
					DataType: entity.FieldTypeVarChar,
					TypeParams: map[string]string{
						"max_length": "65535",
					},
				},
				{
					Name:     "source",
					DataType: entity.FieldTypeVarChar,
					TypeParams: map[string]string{
						"max_length": "255",
					},
				},
				{
					Name:     "embedding",
					DataType: entity.FieldTypeFloatVector,
					TypeParams: map[string]string{
						"dim": "1536", // OpenAI ada-002 embedding dimension
					},
				},
			},
		}

		err = m.client.CreateCollection(ctx, schema, entity.DefaultShardNumber)
		if err != nil {
			log.Printf("Error creating collection: %v", err)
			return false
		}

		// Create index
		idx, err := entity.NewIndexHNSW(entity.L2, 8, 96)
		if err != nil {
			log.Printf("Error creating index: %v", err)
			return false
		}

		err = m.client.CreateIndex(ctx, m.collectionName, "embedding", idx, false)
		if err != nil {
			log.Printf("Error creating index on collection: %v", err)
			return false
		}

		// Load collection
		err = m.client.LoadCollection(ctx, m.collectionName, false)
		if err != nil {
			log.Printf("Error loading collection: %v", err)
			return false
		}
	}

	// For this demo, we'll use dummy embeddings (in a real implementation, you'd generate embeddings using OpenAI's embedding API)
	embeddings := make([][]float32, len(texts))
	for i := range texts {
		// Create dummy embeddings - in real implementation, use OpenAI embeddings API
		embedding := make([]float32, 1536)
		for j := range embedding {
			embedding[j] = float32(i+j) * 0.01 // Simple dummy values
		}
		embeddings[i] = embedding
	}

	// Prepare data for insertion
	log.Printf("📝 Preparing to insert %d documents into collection '%s'", len(texts), m.collectionName)
	textColumn := entity.NewColumnVarChar("text", texts)
	sourceColumn := entity.NewColumnVarChar("source", sources)
	embeddingColumn := entity.NewColumnFloatVector("embedding", 1536, embeddings)

	_, err = m.client.Insert(ctx, m.collectionName, "", textColumn, sourceColumn, embeddingColumn)
	if err != nil {
		log.Printf("❌ Error inserting documents: %v", err)
		return false
	}
	
	log.Printf("✅ Successfully inserted %d documents", len(texts))

	// Flush to ensure data is persisted
	log.Printf("💾 Flushing collection to ensure data persistence...")
	err = m.client.Flush(ctx, m.collectionName, false)
	if err != nil {
		log.Printf("❌ Error flushing collection: %v", err)
		return false
	}
	
	log.Printf("✅ Collection flushed successfully")
	return true
}

func (m *MilvusClientImpl) SearchSimilar(query string, limit int) []Document {
	ctx := context.Background()

	// For this demo, we'll use a dummy query embedding
	// In a real implementation, you'd generate embeddings using OpenAI's embedding API
	queryEmbedding := make([]float32, 1536)
	for i := range queryEmbedding {
		queryEmbedding[i] = float32(i) * 0.01 // Simple dummy values
	}

	searchParams, _ := entity.NewIndexHNSWSearchParam(16)
	results, err := m.client.Search(
		ctx,
		m.collectionName,
		[]string{},
		"",
		[]string{"text", "source"},
		[]entity.Vector{entity.FloatVector(queryEmbedding)},
		"embedding",
		entity.L2,
		limit,
		searchParams,
	)

	if err != nil {
		log.Printf("Error searching documents: %v", err)
		return []Document{}
	}

	var documents []Document
	if len(results) > 0 {
		log.Printf("🔍 Milvus search returned %d results", results[0].ResultCount)
		for i := 0; i < results[0].ResultCount; i++ {
			text, _ := results[0].Fields.GetColumn("text").Get(i)
			source, _ := results[0].Fields.GetColumn("source").Get(i)
			
			// Get similarity score (Milvus returns distance, convert to similarity)
			// For L2 distance, smaller values mean more similar
			distance := results[0].Scores[i]
			// Convert L2 distance to similarity score (0-1 range)
			// Using exponential decay: similarity = e^(-distance)
			similarity := float32(1.0 / (1.0 + distance))
			
			log.Printf("   🎯 Document %d: L2 distance=%.4f, similarity=%.4f (%.1f%%)", 
				i+1, distance, similarity, similarity*100)
			
			documents = append(documents, Document{
				Text:       text.(string),
				Source:     source.(string),
				Similarity: similarity,
			})
		}
	} else {
		log.Printf("⚠️  No documents found matching the query")
	}

	return documents
}

func main() {
	// Check for required environment variables
	openaiAPIKey := os.Getenv("OPENAI_API_KEY")
	if openaiAPIKey == "" {
		log.Println("Warning: OPENAI_API_KEY not set. Using demo mode.")
		runDemoMode()
		return
	}

	milvusHost := os.Getenv("MILVUS_HOST")
	if milvusHost == "" {
		milvusHost = "localhost"
	}

	milvusPort := os.Getenv("MILVUS_PORT")
	if milvusPort == "" {
		milvusPort = "19530"
	}

	collectionName := os.Getenv("COLLECTION_NAME")
	if collectionName == "" {
		collectionName = "rag_documents"
	}

	// Initialize OpenAI client
	openaiClient := &OpenAIClientImpl{
		client: openai.NewClient(openaiAPIKey),
	}

	// Initialize Milvus client
	milvusClient, err := client.NewGrpcClient(context.Background(), fmt.Sprintf("%s:%s", milvusHost, milvusPort))
	if err != nil {
		log.Fatalf("Failed to connect to Milvus: %v", err)
	}
	defer milvusClient.Close()

	milvusClientImpl := &MilvusClientImpl{
		client:         milvusClient,
		collectionName: collectionName,
	}

	// Create RAG engine
	engine := NewRAGEngine(openaiClient, milvusClientImpl)

	// Demo: Add some documents
	log.Println("🚀 Starting RAG Engine Demo")
	log.Println("=" + strings.Repeat("=", 50))
	
	log.Println("📚 Phase 1: Document Ingestion")
	texts := []string{
		"Go is a programming language developed by Google. It's known for its simplicity and efficiency.",
		"Milvus is an open-source vector database that supports similarity search and AI applications.",
		"RAG (Retrieval-Augmented Generation) combines information retrieval with language generation.",
		"Docker is a platform for developing, shipping, and running applications using containerization.",
	}
	sources := []string{
		"Go Documentation",
		"Milvus Documentation", 
		"AI Research Paper",
		"Docker Documentation",
	}

	log.Printf("📄 Preparing to ingest %d documents:", len(texts))
	for i, text := range texts {
		log.Printf("   %d. %s (Source: %s)", i+1, truncateText(text, 60), sources[i])
	}

	success := engine.AddDocuments(texts, sources)
	if !success {
		log.Fatalf("❌ Failed to add documents to the knowledge base")
	}
	log.Println("✅ All documents successfully added to knowledge base!")

	// Demo: Search and generate response
	log.Println("\n🔍 Phase 2: Query Processing & Retrieval")
	log.Println("=" + strings.Repeat("=", 50))
	
	query := "What is Go programming language?"
	log.Printf("❓ User Query: %s", query)
	
	log.Println("\n🎯 Performing vector similarity search...")
	context := milvusClientImpl.SearchSimilar(query, 3)
	log.Printf("📊 Retrieved %d relevant documents from knowledge base", len(context))

	log.Println("\n🤖 Phase 3: Response Generation")
	log.Println("=" + strings.Repeat("=", 50))
	
	response, err := engine.GenerateResponse(query, context, "gpt-3.5-turbo")
	if err != nil {
		log.Fatalf("❌ Failed to generate response: %v", err)
	}

	log.Println("\n📋 Final Results:")
	log.Println("=" + strings.Repeat("=", 50))
	fmt.Printf("❓ Query: %s\n", query)
	fmt.Printf("✅ Response: %s\n", response)
	log.Printf("📈 Processing completed successfully!")
}

// runDemoMode runs the application without OpenAI API, using mock responses
func runDemoMode() {
	fmt.Println("Running in demo mode (no OpenAI API key provided)")
	fmt.Println("This demonstrates the RAG engine structure without actual LLM calls.")

	// Create mock implementations
	mockOpenAI := &mockOpenAIClient{}
	mockMilvus := &mockMilvusClient{
		documents: []Document{
			{Text: "Go is a programming language developed by Google.", Source: "Go Docs", Similarity: 0.85},
			{Text: "Milvus is a vector database for AI applications.", Source: "Milvus Docs", Similarity: 0.72},
		},
	}

	engine := NewRAGEngine(mockOpenAI, mockMilvus)

	// Demo functionality
	fmt.Println("\n1. Adding documents...")
	texts := []string{"Sample document about Go programming"}
	sources := []string{"Demo Source"}
	success := engine.AddDocuments(texts, sources)
	fmt.Printf("Documents added: %t\n", success)

	fmt.Println("\n2. Searching for similar documents...")
	context := mockMilvus.SearchSimilar("What is Go?", 2)
	fmt.Printf("Found %d relevant documents\n", len(context))

	fmt.Println("\n3. Generating response...")
	response, err := engine.GenerateResponse("What is Go?", context, "gpt-3.5-turbo")
	if err != nil {
		log.Printf("Error: %v", err)
		return
	}
	fmt.Printf("Response: %s\n", response)

	fmt.Println("\n4. Testing text chunking...")
	longText := strings.Repeat("This is a sample sentence for chunking. ", 20)
	chunks := ChunkText(longText, 100, 20)
	fmt.Printf("Split text into %d chunks\n", len(chunks))
	for i, chunk := range chunks {
		fmt.Printf("Chunk %d: %s...\n", i+1, chunk[:min(50, len(chunk))])
	}
}

// Mock implementations for demo mode
type mockOpenAIClient struct{}

func (m *mockOpenAIClient) ChatCompletion(model string, messages []Message) (string, error) {
	return "This is a mock response from the RAG engine. In a real implementation, this would be generated by OpenAI's GPT model based on the provided context.", nil
}

type mockMilvusClient struct {
	documents []Document
}

func (m *mockMilvusClient) InsertDocuments(texts, sources []string) bool {
	for i, text := range texts {
		if i < len(sources) {
			// Assign random similarity for demo purposes
			similarity := 0.6 + (float32(i%5) * 0.08) // Values between 0.6 and 0.92
			m.documents = append(m.documents, Document{Text: text, Source: sources[i], Similarity: similarity})
		}
	}
	return true
}

func (m *mockMilvusClient) SearchSimilar(query string, limit int) []Document {
	// Return up to 'limit' documents
	if len(m.documents) <= limit {
		return m.documents
	}
	return m.documents[:limit]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
