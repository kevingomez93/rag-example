package main

import (
	"fmt"
	"log"
	"strings"
)

// Message represents a chat message.
type Message struct {
	Role    string
	Content string
}

// Document holds retrieved text with its source and similarity score.
type Document struct {
	Text       string
	Source     string
	Similarity float32 // Similarity score (0.0 to 1.0, higher is more similar)
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

// GenerateResponse queries the LLM with context and provides detailed logging.
func (r *RAGEngine) GenerateResponse(query string, ctx []Document, model string) (string, error) {
	// Log query details
	log.Printf("🔍 Processing query: %s", query)
	log.Printf("📊 Using %d retrieved documents for context", len(ctx))
	
	// Calculate and log similarity metrics
	if len(ctx) > 0 {
		var totalSimilarity float32
		maxSimilarity := ctx[0].Similarity
		minSimilarity := ctx[0].Similarity
		
		log.Println("📋 Document relevance analysis:")
		for i, doc := range ctx {
			totalSimilarity += doc.Similarity
			if doc.Similarity > maxSimilarity {
				maxSimilarity = doc.Similarity
			}
			if doc.Similarity < minSimilarity {
				minSimilarity = doc.Similarity
			}
			
			// Convert similarity to percentage and relevance category
			percentage := doc.Similarity * 100
			relevance := getRelevanceCategory(doc.Similarity)
			
			log.Printf("   📄 Document %d: %.2f%% similarity (%s)", 
				i+1, percentage, relevance)
			log.Printf("      Source: %s", doc.Source)
			log.Printf("      Preview: %s...", truncateText(doc.Text, 80))
		}
		
		avgSimilarity := totalSimilarity / float32(len(ctx))
		log.Printf("📈 Similarity Statistics:")
		log.Printf("   Average: %.2f%% | Max: %.2f%% | Min: %.2f%%", 
			avgSimilarity*100, maxSimilarity*100, minSimilarity*100)
		
		// Quality assessment
		qualityScore := calculateQualityScore(ctx)
		log.Printf("🎯 Context Quality Score: %.1f/10.0 (%s)", 
			qualityScore, getQualityDescription(qualityScore))
	}

	var contextBuilder strings.Builder
	for i, doc := range ctx {
		contextBuilder.WriteString(fmt.Sprintf("Source %d (%.1f%% relevant): %s\n", 
			i+1, doc.Similarity*100, doc.Source))
		contextBuilder.WriteString("Content: ")
		contextBuilder.WriteString(doc.Text)
		contextBuilder.WriteString("\n\n")
	}
	context := strings.TrimSpace(contextBuilder.String())
	
	prompt := "You are a helpful assistant that answers questions based on the provided context.\n" +
		"Use the context below to answer the user's question. If the answer cannot be found in the context,\n" +
		"say \"I don't have enough information to answer that question based on the provided context.\"\n\n" +
		"Context:\n" + context + "\n\nQuestion: " + query + "\n\nAnswer:"

	log.Printf("🤖 Generating response using model: %s", model)
	messages := []Message{
		{Role: "system", Content: "You are a helpful assistant that answers questions based on provided context."},
		{Role: "user", Content: prompt},
	}
	
	response, err := r.openai.ChatCompletion(model, messages)
	if err != nil {
		log.Printf("❌ Error generating response: %v", err)
		return "", err
	}
	
	log.Printf("✅ Response generated successfully (%d characters)", len(response))
	return response, nil
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

// Helper functions for enhanced logging

// getRelevanceCategory categorizes similarity scores into human-readable terms
func getRelevanceCategory(similarity float32) string {
	switch {
	case similarity >= 0.9:
		return "Excellent Match"
	case similarity >= 0.8:
		return "Very Relevant"
	case similarity >= 0.7:
		return "Relevant"
	case similarity >= 0.6:
		return "Moderately Relevant"
	case similarity >= 0.5:
		return "Somewhat Relevant"
	case similarity >= 0.4:
		return "Low Relevance"
	default:
		return "Poor Match"
	}
}

// truncateText truncates text to a specified length with ellipsis
func truncateText(text string, maxLen int) string {
	if len(text) <= maxLen {
		return text
	}
	return text[:maxLen-3] + "..."
}

// calculateQualityScore calculates an overall quality score for the retrieved context
func calculateQualityScore(docs []Document) float32 {
	if len(docs) == 0 {
		return 0.0
	}
	
	var totalScore float32
	var weights []float32 = []float32{0.5, 0.3, 0.2} // Decreasing weights for ranked results
	
	for i, doc := range docs {
		weight := float32(1.0)
		if i < len(weights) {
			weight = weights[i]
		} else {
			weight = 0.1 // Very low weight for documents beyond top 3
		}
		
		// Score based on similarity with positional weighting
		score := doc.Similarity * weight * 10.0
		totalScore += score
	}
	
	// Normalize to 0-10 scale
	maxPossibleScore := float32(0.5 + 0.3 + 0.2) * 10.0 // Assuming perfect similarity
	if len(docs) == 1 {
		maxPossibleScore = 5.0
	} else if len(docs) == 2 {
		maxPossibleScore = 8.0
	}
	
	qualityScore := (totalScore / maxPossibleScore) * 10.0
	if qualityScore > 10.0 {
		qualityScore = 10.0
	}
	
	return qualityScore
}

// getQualityDescription provides a human-readable description of the quality score
func getQualityDescription(score float32) string {
	switch {
	case score >= 9.0:
		return "Exceptional"
	case score >= 8.0:
		return "Excellent"
	case score >= 7.0:
		return "Very Good"
	case score >= 6.0:
		return "Good"
	case score >= 5.0:
		return "Fair"
	case score >= 4.0:
		return "Below Average"
	case score >= 3.0:
		return "Poor"
	default:
		return "Very Poor"
	}
}
