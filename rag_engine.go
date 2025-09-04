package rag

import (
	"strings"
)

// Message represents a chat message.
type Message struct {
	Role    string
	Content string
}

// Document holds retrieved text with its source.
type Document struct {
	Text   string
	Source string
}

// OpenAIClient defines the minimal interface we need for chat completions.
type OpenAIClient interface {
	ChatCompletion(model string, messages []Message) (string, error)
}

// MilvusClient defines the minimal interface for document storage and retrieval.
type MilvusClient interface {
	InsertDocuments(texts, sources []string) bool
	SearchSimilar(query string, limit int) []Document
}

// RAGEngine ties together the LLM and vector database clients.
type RAGEngine struct {
	openai OpenAIClient
	milvus MilvusClient
}

// NewRAGEngine builds a new engine with provided dependencies.
func NewRAGEngine(openai OpenAIClient, milvus MilvusClient) *RAGEngine {
	return &RAGEngine{openai: openai, milvus: milvus}
}

// AddDocuments inserts documents into the vector store.
func (r *RAGEngine) AddDocuments(texts, sources []string) bool {
	if len(texts) != len(sources) {
		return false
	}
	return r.milvus.InsertDocuments(texts, sources)
}

// GenerateResponse queries the LLM with context.
func (r *RAGEngine) GenerateResponse(query string, ctx []Document, model string) (string, error) {
	var contextBuilder strings.Builder
	for _, doc := range ctx {
		contextBuilder.WriteString("Source: ")
		contextBuilder.WriteString(doc.Source)
		contextBuilder.WriteString("\nContent: ")
		contextBuilder.WriteString(doc.Text)
		contextBuilder.WriteString("\n\n")
	}
	context := strings.TrimSpace(contextBuilder.String())
	prompt := "You are a helpful assistant that answers questions based on the provided context.\n" +
		"Use the context below to answer the user's question. If the answer cannot be found in the context,\n" +
		"say \"I don't have enough information to answer that question based on the provided context.\"\n\n" +
		"Context:\n" + context + "\n\nQuestion: " + query + "\n\nAnswer:"

	messages := []Message{
		{Role: "system", Content: "You are a helpful assistant that answers questions based on provided context."},
		{Role: "user", Content: prompt},
	}
	return r.openai.ChatCompletion(model, messages)
}

// ChunkText splits text into overlapping chunks.
func ChunkText(text string, chunkSize, overlap int) []string {
	var chunks []string
	start := 0
	for start < len(text) {
		end := start + chunkSize
		if end > len(text) {
			end = len(text)
		}
		chunk := text[start:end]

		if end < len(text) {
			lastPeriod := strings.LastIndex(chunk, ".")
			lastNewline := strings.LastIndex(chunk, "\n")
			lastBreak := lastPeriod
			if lastNewline > lastBreak {
				lastBreak = lastNewline
			}
			if lastBreak > start+chunkSize/2 {
				chunk = chunk[:lastBreak+1]
				end = start + len(chunk)
			}
		}

		chunk = strings.TrimSpace(chunk)
		if chunk != "" {
			chunks = append(chunks, chunk)
		}
		if end == len(text) {
			break
		}
		start = end - overlap
		if start < 0 {
			start = 0
		}
	}
	return chunks
}
