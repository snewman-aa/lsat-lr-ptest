import pytest

from clients.gemini_client import GeminiClient, SampleQuestion


class DummyResponse:
    def __init__(self, text, parsed):
        self.text   = text
        self.parsed = parsed


class DummyModels:
    def generate_content(self, model, contents, config):
        assert config.response_schema is SampleQuestion
        sq = SampleQuestion(
            stimulus="s", prompt="p",
            answers=["A","B","C","D","E"],
            correct_answer="A",
            explanation="e"
        )
        return DummyResponse(text=sq.model_dump_json(), parsed=sq)


class DummyClient:
    def __init__(self, api_key):
        self.models = DummyModels()


@pytest.fixture(autouse=True)
def patch_genai(monkeypatch):
    import google.genai as genai_mod
    monkeypatch.setattr(genai_mod, "Client", lambda api_key: DummyClient(api_key))


def test_generate_sample_question(monkeypatch):
    gc = GeminiClient()
    sq = gc.generate_sample_question("foo")
    assert isinstance(sq, SampleQuestion)
    assert sq.prompt == "p"
