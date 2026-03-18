[🇷🇺 Русская версия](README_RU.md)

# LLM with RAG via YandexCloudML

The initial integration involved preprocessing knowledge base texts from training video and masterclass transcriptions into a Q&A markdown format, uploading files to server storage, building a search index, and configuring an AI agent with knowledge base search tools and Internet queries when needed. The corporate chat, upon receiving a user query, would forward it to the AI agent and relay the response.
The knowledge base was regularly updated by company experts.

Folder contents:
- **.md, .jsonl** — knowledge base files;
- **zapros_rag.py** — query script;
- **transform_jsonl_md.py** — script for converting texts/json into Q&A markdown format.
