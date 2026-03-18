[🇷🇺 Русская версия](README_RU.md)

# Payment Classification by Universal Categories

In this section we add a universal categories feature based on the Russian Ministry of Justice classification and recommended industry practices, applying data preprocessing and labeling methods, and training a set of models for classifying incoming payments of client funds.

Stages:
- payment data is loaded via API in a format similar to incoming payment forecasting (request_3.sql), but supplemented with article names, projects, and donors (request_4.sql), as well as UC labels (request_5.sql) with the incoming parameter;
- uc_data_labeling — labeling payments by universal categories according to an agreed-upon list (универсальные категории_3_1.docx/articles_uc.csv/articles_uc_added.csv) and partially manually;
- gpt_classification — testing universal category classification using public LLM predictions — ChatGPT 5/Gemini 2.5 pro/YandexGPT and evaluating metrics.
- own_classification — building and testing custom ML models for payment classification by universal categories.
