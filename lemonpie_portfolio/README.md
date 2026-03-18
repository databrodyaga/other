[🇷🇺 Русская версия](README_RU.md)

# ML Section of the LemonPie Project

This section contains machine learning models and scripts for various tasks within the LemonPie system.

## Project Structure

### `forecast_payments/`
Models for forecasting incoming payments for nonprofit organizations (NGOs).

- **first_try/** - initial experiments with various models (SARIMAX, LSTM, CatBoost)
- **second_try(real_data)/** - working with real-world data
  - **EDA/** - exploratory data analysis, feature selection
  - **training_models/** - model training for payment forecasting

### `universal_categories/`
Payment classification by universal categories according to the Russian Ministry of Justice classification.

- **uc_data_labeling/** - data labeling for training
- **gpt_classification/** - testing classification using public LLMs (ChatGPT, Gemini, YandexGPT)
  - **ver_1** - version 1
  - **ver_2-ver_8** - subsequent versions with improvements
- **own_classification/** - custom ML classification models
  - **train_test/** - model training experiments
  - **prod/** - production version of the model

### `yandex_ai_studio_rag/`
LLM with RAG (Retrieval-Augmented Generation) setup for the technical support chat.

- **rag_ver_1/** - first version based on the YandexCloudML library
- **rag_ver_2/** - second version based on the Responses API (OpenAI-compatible interface)

## Technologies

- **Python** - primary development language
- **Pandas, NumPy** - data processing
- **Scikit-learn** - classical ML models
- **CatBoost, XGBoost** - gradient boosting
- **Transformers (Hugging Face)** - BERT models for text
- **YandexGPT, ChatGPT, Gemini** - LLMs for classification
- **Jupyter Notebooks** - exploratory analysis and experiments

## Key Tasks

1. **Payment Forecasting** - predicting incoming payments based on historical data
2. **Universal Category Classification** - automated payment categorization
3. **RAG for Tech Support** - intelligent assistant powered by a knowledge base

## Important Notes

- All training data files contain only examples (single row)
- To work with the models, you need to set up the environment and install dependencies
- API tokens in the code are replaced with placeholders (`YOUR_API_TOKEN`, `YOUR_YANDEX_CLOUD_TOKEN`, `YOUR_FOLDER_ID`). In production, credentials are passed via environment variables (`os.environ`)

## Installing Dependencies

```bash
pip install pandas numpy scikit-learn catboost transformers torch
```

For Yandex Cloud ML:
```bash
pip install yandex-cloud-ml-sdk
```

For the OpenAI-compatible API:
```bash
pip install openai
```
