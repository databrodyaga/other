[🇷🇺 Русская версия](README_RU.md)

# Payment EDA

Analyzing donor payment data to nonprofit funds:

- preprocessing and EDA of the real data export from the database,
- aggregating individual payments to daily level for next-day forecasting,
- generating and adding features,
- checking correlations.

Files in this folder:
 - **second_try_eda.ipynb** — analysis notebook
 - **request_1.sql** — database query (raw dataset **data_1.csv** not uploaded due to size)
 - **users_actual.csv** — active user_id (funds) for raw data filtering
 - **data_with_purpose_mean.parquet**, **lang_model_test.parquet**, **phik_correlation_matrix_full.parquet** — cache files with language model embedding calculations for text field encoding and correlation tables.

Additional research and testing stages:
- **1_2model_vs_3model_choice** — comparison of two training pipelines (with and without time component)
- **2_funds_user_id_choice** — threshold analysis for fund selection for forecasting
- **3_feature_selection** — feature selection for the final model
