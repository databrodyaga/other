# LangGraph + LangChain Learning Project

A collection of small projects created while learning LangGraph, LangChain, and LLM-based applications.

### 1. Chatbot

A minimal asynchronous chatbot built with LangGraph to explore graph-based conversational workflows and create a reusable LLM wrapper.

### 2. Job Description Classifier

A LangGraph workflow that classifies job descriptions using an LLM.

For each description it predicts:

- Job type
  - permanent job
  - project job
- Profession category
- Search type
  - looking for work
  - looking for employee

A single classification node predicts all three fields in one structured LLM call and updates the shared graph state.

### Tested models

- Gemma (Ollama)
- Llama (Groq API)
- Gemini (Google AI)

The implementation is provider-agnostic, making it easy to switch between different LLM backends.

### Final workflow

The final notebook runs one graph over the complete input file:

```text
load jobs (tool) -> select job -> classify
                         ^           |  |
                         |           |  +-- retry
                         |           v
                         +-- next -- collect result
                                      |
                                      v
                             save results (tool) -> END
```
It demonstrates:
- shared typed state;
- async LLM calls and structured output;
- tool-backed input and output nodes;
- conditional edges, a batch loop, and bounded retries;
- local or optional MCP filesystem storage;
- switching Ollama, Groq, and Gemini without changing graph logic.

The storage tools are selected by the deterministic graph. They are intentionally not bound to the LLM: this example demonstrates tool-backed workflow nodes, not an autonomous tool-calling agent.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
