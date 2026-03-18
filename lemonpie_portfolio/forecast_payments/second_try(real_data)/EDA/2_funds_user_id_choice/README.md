[🇷🇺 Русская версия](README_RU.md)

# Fund Selection

Analyzing metrics across all funds (based on the test set) to determine a threshold for selecting funds worth training models and making predictions for: if data is too scarce, models either produce errors (empty data batches when splitting the training set) or perform worse than a naive forecast.

To do this, we built a Python script to run training of the same pipeline across all funds, with selection based on three parameters:
- absolute number of payments over the entire period,
- number of days with payments (active days),
- payment-to-day ratio (all days including zero-activity gaps — to assess the average number of payments over the entire operating period),
- combinations of the above parameters.

**Conclusion**:
The best value for the primary RMSE metric (lower average across all funds is better) was achieved with a threshold of at least 150 active days (excluding zero-activity days); we keep this threshold.


Files in this folder:
 - **second_try_3_models.py** — training and prediction script
 - **request_3.sql** — database query for data export
 - **users_actual.csv** — active user_id (funds) for raw data filtering
 - **2try_data_with_purpose_mean.parquet** — cache files with language model embedding calculations for text field encoding and correlation tables.
 - **comparison...csv** — metric files for different thresholds
 - **comparison.xlsx** — summary table with metrics based on the source CSV files (see above)
