[🇷🇺 Русская версия](README_RU.md)

# Universal Category Classification Using Public LLM Predictions

Collected prediction evaluations from different LLMs with and without fine-tuning.
Each version contains the following key files:

*Промт для классификации.docx* — not present in all versions, as later versions did not use standard GPT interfaces
*gpt_dataset.ipynb* — dataset splitting and preprocessing
**gpt_metrics_5000.ipynb** — **metrics calculation** based on test sample size (main file with comments and metrics)
*yandexgpt_classify.ipynb* — predictions from standard YandexGPT
*yandexgpt_ft_classify.ipynb* — predictions from fine-tuned YandexGPT
*X_test_balanced.csv, y_test_balanced.csv,*
*X_train.csv, y_train.csv,*
*X_test.csv, y_test.csv* — datasets (versions with "_1", "_id" were used for synchronization and verification with later payment batches by id)
*y_pred_chatgpt.csv* — ChatGPT predictions
*y_pred_gemini.csv* — Google Gemini predictions
*y_pred_ygpt.csv* — YandexGPT predictions
*y_pred_ygpt_ft.csv* — fine-tuned YandexGPT predictions
**gpt_classification_prod.py** — production script for data loading, label generation, and uploading to the service database.

Testing stages:
**ver_1** — incorrectly fine-tuned models on 5000 examples of a single class, low effectiveness;
**ver_2** — correctly fine-tuned models using ONLY payment purpose texts, metrics: chatgpt — 75%, yagpt — 70%;
**ver_3** — attempt to fine-tune on an even more class-imbalanced dataset, failed, low metrics;
**ver_4** — testing yagpt fine-tuned for version 2 on a different test set — metrics confirmed;
**ver_5** — fine-tuned LLM on data with added article names, projects, and donors, tested on the same set as before — the LLM simply memorized the distribution based on supplementary names, resulting in 100% metrics;
**ver_6** — 50% of the test dataset (half of examples per class) was modified (added typos, analogies, omissions) — distortions are significant yet yagpt still showed 96% metrics;
**ver_7** — tested the version 5 model on the original test dataset with only payment purpose texts — metrics dropped to 30% — without additional data the model cannot cope;
**ver_8** — tested the version 5 model on a real payment dataset with new article values the model had never seen, 1721 examples with class imbalance, from tests similar to tests 4 and 6 of the custom model. Results are interesting but not outstanding.
