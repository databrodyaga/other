# LangGraph + LangChain Journey

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

Each prediction step is implemented as an individual LangGraph node that updates the shared graph state.

### Tested models

- Qwen (Ollama)
- Llama (Groq API)
- Gemini (Google AI)

The implementation is provider-agnostic, making it easy to switch between different LLM backends.