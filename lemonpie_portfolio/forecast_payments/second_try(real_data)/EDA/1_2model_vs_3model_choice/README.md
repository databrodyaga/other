[🇷🇺 Русская версия](README_RU.md)

# Pipeline Comparison and Selection

Preprocessing data and building two pipelines with 2 models (without time component) and 3 models (with time component) for payment forecasting.
The metrics comparison file shows that the more complete 3-model pipeline with reconstructed time series significantly outperforms the simpler version; we keep the more complex one.

Files in this folder:
 - **second_try_2_models.ipynb** — notebook with 2-model pipeline (without time component)
 - **second_try_3_models.ipynb** — notebook with 3-model pipeline (with time component)
 - **comparison_.csv**, **comparison_.xlx** — metrics comparison of the two pipelines (see above) across available funds
 - **request_3.sql** — database query (raw dataset **data_3.csv** not uploaded due to size)
 - **users_actual.csv** — active user_id (funds) for raw data filtering
 - **2try_data_with_purpose_mean.parquet** — cache file with language model embedding calculations for text field encoding.
