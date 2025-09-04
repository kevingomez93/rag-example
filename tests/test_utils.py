
def test_chunk_text_overlaps(rag):
    text = "A" * 15
    chunks = rag.chunk_text(text, chunk_size=10, overlap=2)
    assert chunks == ["A" * 10, "A" * 7]

