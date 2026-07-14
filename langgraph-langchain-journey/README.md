## LangGraph + LangChain Journey

A small project created to explore the basics of LangGraph, LangChain, and LLM-powered workflows.

### Project

The application classifies job descriptions using an LLM and a LangGraph workflow.

For each input description it predicts:

* Job type
    * permanent job
    * project job
* Profession category
* Search type
    * looking for work
    * looking for employee

### Workflow

```text

Description
    │
    ▼
Job Type
    │
    ▼
Category
    │
    ▼
Search Type
    │
    ▼
Result

```

Each step is implemented as an individual LangGraph node that updates the shared graph state.

### Tested models

* Local Qwen (Ollama)
* Groq API (Llama)
* Google Gemini API

The implementation was intentionally kept provider-agnostic, making it easy to switch between different LLM backends.

Rather than treating knowledge of a specific framework as a prerequisite, I wanted to see how quickly a working solution could be built by learning the fundamentals and applying them in practice.