[🇷🇺 Русская версия](README_RU.md)

# Universal Category Classification Using a Custom ML Model

In this section we analyze data for label prediction, test different data processing pipelines, and configure prediction models.

In this folder:
- check_date_wo_forecasts.ipynb — label availability after manual labeling of the training dataset (all payments through 09.10.2025 inclusive)
- classify_manual.ipynb — manual label upload to the database based on expert/client feedback
- EDA_for_classify.ipynb — data analysis and feature correlations
- prod — payment classification script, production version of the model and supplementary data
- train_test — model selection and data preprocessing pipelines, model/feature tests, production model retraining
