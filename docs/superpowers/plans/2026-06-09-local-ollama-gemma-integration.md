# Local Ollama Gemma 4 Coder Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate Ollama local backend support in the LLM coding system to run Gemma 4 models locally.

**Architecture:** Create an `OllamaBackend` that communicates via HTTP POST requests to a local Ollama instance (`http://localhost:11434/api/chat`), and configure the LLM coder pool to run four instances of `gemma4:e4b` under differing seeds.

**Tech Stack:** Python 3.11+, `httpx`, Ollama HTTP API

---

### Task 1: Create Unit Test for Ollama Backend

**Files:**
- Create: `tests/test_llm_coder_ollama.py`

- [ ] **Step 1: Write mock test for OllamaBackend**
  
  Write a test that mocks `httpx.Client.post` to verify that `OllamaBackend` formats request payloads correctly and correctly parses the binary (0/1) response from Ollama.
  
  ```python
  import httpx
  import pytest
  from unittest.mock import MagicMock
  from david.ingest.llm_coder import LlmCoderConfig, get_backend, OllamaBackend

  def test_ollama_backend_coding(monkeypatch):
      # Mock the HTTP response from Ollama
      mock_response = MagicMock()
      mock_response.status_code = 200
      mock_response.json.return_mock = MagicMock(return_value={
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
      monkeypatch.setattr(httpx.Client, "get", mock_get)

      backend = get_backend(cfg)
      assert isinstance(backend, OllamaBackend)
      
      res = backend.code_item("Tobacco advert on TV", "SIO")
      assert res == 1
  ```

- [ ] **Step 2: Run test to verify it fails**
  
  Run: `uv run pytest tests/test_llm_coder_ollama.py -v`
  Expected: FAIL (ImportError or NotImplementedError since `ollama` provider is not implemented yet)

- [ ] **Step 3: Commit initial test skeleton**
  
  ```bash
  git add tests/test_llm_coder_ollama.py
  git commit -m "test: add failing mock test for Ollama backend"
  ```

---

### Task 2: Implement Ollama Backend in `llm_coder.py`

**Files:**
- Modify: `david/ingest/llm_coder.py`

- [ ] **Step 1: Write OllamaBackend implementation**
  
  Add `OllamaBackend` to `david/ingest/llm_coder.py`:
  
  ```python
  class OllamaBackend:
      """Ollama backend for local binary (0/1) evidence coding.
      
      Communicates with local Ollama service running at http://localhost:11434.
      """

      _SYSTEM = (
          "You are a political-events coder. "
          "For each piece of text, respond with exactly 0 or 1 — "
          "1 if the text contains evidence of the stated tactic, 0 if not. "
          "Respond with a single digit only."
      )

      def __init__(self, cfg: LlmCoderConfig) -> None:
          self._cfg = cfg
          self._url = "http://localhost:11434/api/chat"
          self._template = AnthropicBackend._load_template(cfg.prompt_template_id)
          
          # Probe to check if Ollama is running
          import httpx
          try:
              resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
              if resp.status_code != 200:
                  raise RuntimeError(f"Ollama returned status code {resp.status_code}")
          except Exception as e:
              raise RuntimeError(
                  f"Ollama local service does not appear to be running. "
                  f"Please start Ollama before executing. Error: {e}"
              )

      def code_item(self, text: str, tactic_class: str) -> int:
          """Return 1 if `text` contains evidence of `tactic_class`, else 0."""
          import httpx
          user_msg = self._template.format(
              tactic_class=tactic_class,
              text=text[:4000],
          )
          payload = {
              "model": self._cfg.model,
              "messages": [
                  {"role": "system", "content": self._SYSTEM},
                  {"role": "user", "content": user_msg},
              ],
              "options": {
                  "temperature": self._cfg.temperature,
                  "seed": self._cfg.seed,
              },
              "stream": False
          }
          try:
              with httpx.Client(timeout=30.0) as client:
                  resp = client.post(self._url, json=payload)
                  if resp.status_code == 200:
                      raw = resp.json().get("message", {}).get("content", "").strip()
                      if raw.startswith("1") or raw.lower().startswith("yes"):
                          return 1
                  return 0
          except Exception:
              return 0
  ```

- [ ] **Step 2: Register `"ollama"` in `get_backend()`**
  
  Modify `_PROVIDER_DISPATCH` in `get_backend()` inside `david/ingest/llm_coder.py`:
  
  ```python
      _PROVIDER_DISPATCH = {
          "anthropic": AnthropicBackend,
          "groq": GroqBackend,
          "ollama": OllamaBackend,
      }
  ```

- [ ] **Step 3: Run pytest to verify it passes**
  
  Run: `uv run pytest tests/test_llm_coder_ollama.py -v`
  Expected: PASS

- [ ] **Step 4: Commit implementation**
  
  ```bash
  git add david/ingest/llm_coder.py
  git commit -m "feat: implement local Ollama backend for LLM coding"
  ```

---

### Task 3: Update LLM Pool Configuration

**Files:**
- Modify: `config/llm_pool.json`

- [ ] **Step 1: Replace configurations in `config/llm_pool.json`**
  
  Update file content to define 4 local Gemma 4 E4B coders (seeds 1 to 4).

- [ ] **Step 2: Commit configuration changes**
  
  ```bash
  git add config/llm_pool.json
  git commit -m "config: swap cloud coders for local Gemma 4 E4B Ollama configurations"
  ```
