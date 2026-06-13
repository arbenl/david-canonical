import httpx
import pytest
from unittest.mock import MagicMock
from david.ingest.llm_coder import LlmCoderConfig, get_backend, OllamaBackend

def test_ollama_backend_coding(monkeypatch):
    # Mock the HTTP response from Ollama
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value={
        "message": {"content": "1"}
    })
    
    def mock_post(*args, **kwargs):
        # Verify payload format
        json_data = kwargs.get("json", {})
        assert json_data["model"] == "gemma4:e4b"
        assert json_data["stream"] is False
        assert json_data["options"]["seed"] == 42
        return mock_response

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    cfg = LlmCoderConfig(
        coder_id="test_ollama",
        provider="ollama",
        model="gemma4:e4b",
        prompt_template_id="default",
        seed=42,
        temperature=0.0
    )
    
    # We mock the probe to return 200 OK
    def mock_get(*args, **kwargs):
        mock_probe = MagicMock()
        mock_probe.status_code = 200
        return mock_probe
    monkeypatch.setattr(httpx, "get", mock_get)

    backend = get_backend(cfg)
    assert isinstance(backend, OllamaBackend)
    
    res = backend.code_item("Tobacco advert on TV", "SIO")
    assert res == 1
