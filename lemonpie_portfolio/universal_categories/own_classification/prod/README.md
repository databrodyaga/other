[🇷🇺 Русская версия](README_RU.md)

# Production Model and Payment Classification Script

In this folder:
- classify_forecast_prod (ipynb/py) — jupyter/python script for loading new payments, generating predictions, and uploading labels to the database
- classify_forecast_prod.log — script execution log
- model_cb.cbm — CatBoost model for generating predictions
- ipca_*.pkl, scaler_*.pkl — supplementary data for generating model features from text fields of the source data; these representations are generated during model (re)training
