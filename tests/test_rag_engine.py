
def test_add_documents_mismatched_lengths(rag):
    engine = rag.RAGEngine(openai_api_key="test")
    assert not engine.add_documents(["doc1"], ["s1", "s2"])


def test_generate_response_uses_context(rag):
    engine = rag.RAGEngine(openai_api_key="test")
    context = [{"text": "info about cats", "source": "src"}]
    response = engine.generate_response("question?", context, model="gpt-test")
    assert response == "stubbed"
    comps = engine.openai_client.chat.completions
    assert "info about cats" in comps.last_kwargs["messages"][1]["content"]
    assert comps.last_kwargs["model"] == "gpt-test"

