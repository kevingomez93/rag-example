# RAG Chat with Milvus (Go)

This repository provides a minimal Retrieval-Augmented Generation (RAG) engine written in Go. The engine links an OpenAI chat model with a Milvus vector database.

## Features
- Document insertion with source tracking
- Retrieval of relevant context from Milvus
- Chat completion through a pluggable OpenAI client
- Text chunking with overlap for improved context windows
- Unit tests with stubbed dependencies

## Getting Started

### Prerequisites
- Go 1.20+ (tested with Go 1.24)
- An OpenAI API key
- Running Milvus instance (see `docker-compose.yml` for local setup)

### Run Tests
```bash
go test -v
```

### Using the Engine
Implement the `OpenAIClient` and `MilvusClient` interfaces defined in `rag_engine.go` and pass them to `NewRAGEngine`:

```go
package main

import (
    "fmt"

    "rag-example"
)

func main() {
    oa := &MyOpenAI{}
    mv := &MyMilvus{}
    engine := rag.NewRAGEngine(oa, mv)

    texts := []string{"Example document"}
    sources := []string{"Doc 1"}
    engine.AddDocuments(texts, sources)

    ctx := mv.SearchSimilar("question", 3)
    resp, _ := engine.GenerateResponse("question", ctx, "gpt-4o")
    fmt.Println(resp)
}
```

See `rag_engine_test.go` for additional usage examples with mock implementations.

## Milvus Setup

To launch a local Milvus instance for development:

```bash
docker-compose up -d
```

Environment variables used by the engine are illustrated in `env_example.txt`.
