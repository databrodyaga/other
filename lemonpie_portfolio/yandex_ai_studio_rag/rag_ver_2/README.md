[🇷🇺 Русская версия](README_RU.md)

# LLM with RAG via Responses API (OpenAI)

Current integration version in preparation for the Yandex Cloud AI transition to the widely adopted format for AI agents and RAG tool integration. Changes in this version:
- knowledge base text preprocessing schema changed — in addition to uploading raw training video transcriptions, we tested and adopted LLM-based preprocessing into the recommended *.jsonl schema `{ "body": "text" }` for more effective knowledge base search;
- in addition to working with knowledge base files and building search indexes via the web interface, management scripts were prepared for all operations directly via API, especially since efficient handling of jsonl files is only implemented this way;
- the LLM query was rebuilt to account for the API architecture changes, with added diagnostics for connected RAG tools (file_search, web_search);
- the chat-LLM interaction flow remained unchanged; the knowledge base is regularly updated by company experts.

Folder contents:
- **.md, .jsonl** — knowledge base files;
- **convert_jsonl_to_body.py** — script for converting free-form jsonl to the recommended `{ "body": "text" }` schema;
- **index_manager.py** — script for managing knowledge base files and the search index (upload/delete files of various types, generate/delete index);
- **upload_jsonl_to_index.py** — simplified script for uploading recommended-schema jsonl files to the search index (subset of the general manager functionality, see above);
- **zapros_rag_2.py** — LLM query script.
