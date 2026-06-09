# Design Spec: Local Ollama Gemma 4 Coder Integration

## Goal Description
Integrate local LLM coding support via Ollama, specifically utilizing Google's Gemma 4 models (e.g. `gemma4:e4b`). This removes dependency on external APIs (Groq and Anthropic) for the tobacco interference classification engine, providing:
1. Zero cost and zero API rate-limit throttling.
2. Complete scientific reproducibility by freezing local model weights.
3. Offline execution support.

## Proposed Changes

### Ingest Coder Backend

#### [MODIFY] [llm_coder.py](file:///Users/arbenlila/development/david/canonical/david/ingest/llm_coder.py)
* Create the `OllamaBackend` class that sends synchronous HTTP POST requests to `http://localhost:11434/api/chat`.
* Check if Ollama is running during initial instantiation and raise a descriptive error if not found.
* Implement error handling that falls back to returning `0` if connection issues occur during execution.
* Register `"ollama"` in `_PROVIDER_DISPATCH`.

### Registry Configuration

#### [MODIFY] [llm_pool.json](file:///Users/arbenlila/development/david/canonical/config/llm_pool.json)
* Replace existing Groq and Anthropic configurations with four distinct `ollama` coder setups using `gemma4:e4b` with unique seeds.

## Verification Plan

### Automated Verification
* Verify that the local Ollama API is running.
* Run unit tests to check `OllamaBackend` logic.

### Manual Verification
* Install Ollama and pull `gemma4:e4b`.
* Run the ingest CLI command `david ingest` locally.
* Compare processing speed and labels coded with those from the previous cloud API runs.
