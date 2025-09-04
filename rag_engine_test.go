package main

import (
	"strings"
	"testing"
)

type dummyOpenAI struct {
	lastModel    string
	lastMessages []Message
}

func (d *dummyOpenAI) ChatCompletion(model string, messages []Message) (string, error) {
	d.lastModel = model
	d.lastMessages = messages
	return "stubbed", nil
}

type dummyMilvus struct {
	insertedTexts   []string
	insertedSources []string
	lastQuery       string
	lastLimit       int
}

func (d *dummyMilvus) InsertDocuments(texts, sources []string) bool {
	d.insertedTexts = texts
	d.insertedSources = sources
	return true
}

func (d *dummyMilvus) SearchSimilar(query string, limit int) []Document {
	d.lastQuery = query
	d.lastLimit = limit
	return nil
}

func TestAddDocumentsMismatchedLengths(t *testing.T) {
	oa := &dummyOpenAI{}
	mv := &dummyMilvus{}
	engine := NewRAGEngine(oa, mv)
	if engine.AddDocuments([]string{"doc1"}, []string{"s1", "s2"}) {
		t.Fatalf("expected AddDocuments to fail on mismatched lengths")
	}
}

func TestGenerateResponseUsesContext(t *testing.T) {
	oa := &dummyOpenAI{}
	mv := &dummyMilvus{}
	engine := NewRAGEngine(oa, mv)
	ctx := []Document{{Text: "info about cats", Source: "src", Similarity: 0.85}}
	resp, err := engine.GenerateResponse("question?", ctx, "gpt-test")
	if err != nil {
		t.Fatalf("GenerateResponse returned error: %v", err)
	}
	if resp != "stubbed" {
		t.Fatalf("unexpected response: %s", resp)
	}
	if oa.lastModel != "gpt-test" {
		t.Fatalf("model not passed to openai client")
	}
	if len(oa.lastMessages) < 2 || !strings.Contains(oa.lastMessages[1].Content, "info about cats") {
		t.Fatalf("context not passed to openai client")
	}
}

func TestChunkTextOverlaps(t *testing.T) {
	text := strings.Repeat("A", 15)
	chunks := ChunkText(text, 10, 2)
	expected := []string{strings.Repeat("A", 10), strings.Repeat("A", 7)}
	if len(chunks) != len(expected) {
		t.Fatalf("expected %d chunks, got %d", len(expected), len(chunks))
	}
	for i := range expected {
		if chunks[i] != expected[i] {
			t.Fatalf("chunk %d expected %s got %s", i, expected[i], chunks[i])
		}
	}
}
