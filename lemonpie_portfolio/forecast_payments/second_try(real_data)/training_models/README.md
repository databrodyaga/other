[🇷🇺 Русская версия](README_RU.md)

# Model Training and Production Script

Refactoring the Python script from the ipynb notebook EDA/3_feature_selection.
Preprocessing data and training models for payment forecasting.
The current script **second_try.py** is effectively the production model for daily predictions.

Files in this folder:
 - **second_try.py** — final script with the 3-model pipeline (with time component)
 - **request_5.sql** — database query (raw dataset **data_*.csv** not uploaded due to size)
 - **users_actual.csv** — active user_id (funds) for raw data filtering, not used at this stage
