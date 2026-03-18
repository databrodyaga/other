[🇷🇺 Русская версия](README_RU.md)

# CatBoost Tuning for Monthly Payment Forecasting

In this section we tune fund selection parameters for forecasting, since only a single CatBoost model remains instead of the ensemble and it is less demanding on the number of periods (unlike SARIMAX). We also select the number of hyperparameter sets for the most effective forecasting of payments by individual funds (each fund gets its own trained model).

Dataset for stages 1–3: all payments up to 01.01.2026 (preprocessing is identical in all scripts).

In **stage one** — to verify the effectiveness of fund selection parameter tuning, we added median-based metric calculations as a naive forecast for funds with insufficient data for model-based forecasting. If the model-based forecast for a fund produces smaller errors than a simple median from the previous period, then the model is more effective for that fund.
Initial selection parameters were:
 - payments present in the last 3 months → changed to 4 months;
 - at least 18 months total with payments → changed to 6 months;
 - a filter on starting values of the series — requiring at least 3 non-zero values in the first 6-month window, with at least 2 such windows, sliding from the first value until the condition is met (the "starting" tail is discarded) — thresholds changed to 3-1-1.

**Stage conclusions**:
Metrics for the "old" 23 funds remained absolutely identical (differences within rounding error). This confirms that adding new funds does not "break" training for existing ones, since models are built independently.
New fund effectiveness: The main success criterion is catboost_gain (difference between Median error and CatBoost error). A positive value means the model outperforms the naive forecast.
Successful cases (Model > Median):
Before (out of 23 funds):
	- CatBoost won: 16 funds (~70%)
	- Median won: 7 funds (~30%)
After (out of 63 funds):
	- CatBoost won: 47 funds (~75%)
	- Median won: 16 funds (~25%).

When expanding the funnel (from 23 to 63 funds), the CatBoost win rate even increased (from 70% to 75%). This suggests that among previously filtered-out funds, there were many whose behavior CatBoost predicts much better than a simple median. These are likely growing or changing funds where the median (focused on the past) systematically errs, while boosting captures the dynamics.
Given the above, loosening the filters is fully justified. Although the model loses to the median on some "sparse" series (like 1110), the aggregate gain on large funds (722, 1054) far outweighs local failures. We started forecasting revenue where previously the fund was simply ignored.

In **stage two**, for the expanded fund set we added a hyperparameter grid to better adapt to more funds (while controlling overfitting). Specifically: added depth 1 (very simple trees), increased L2 regularization to 10, added learning_rate variations.
**Stage conclusion**: The expanded grid works more effectively than the baseline in 70–80% of cases. It particularly helps funds with medium history (like id 176), finding simpler and more robust models for them (e.g., depth 1).
The only serious "outlier" is fund 722, but on average (by total RMSE across all funds) the expanded grid wins.

In **stage three**, we discovered that the script was incorrectly aggregating monthly data (effectively by the first day of the month, not by the sum of all monthly payments). After fixing the bug and retraining on the same data up to 01.01.2026, we recalculated metrics — comparison..ver_6 — with the same hyperparameters as stage 2, and comparison..ver_7 with hyperparameters tuned on monthly data. Result: comparison..ver_6 — metrics improved on correct monthly data due to greater stability of monthly sums; with the expanded hyperparameter set in comparison..ver_7, improvement was observed for most funds.


Files in this folder:
 - **comparison...act_restrictions.csv** — metrics with original selection parameters, tested on the last two months (23 funds selected for model-based forecasting);
 - **comparison...new_restrictions.csv** — metrics with loosened selection parameters, tested on the last two months (63 funds selected for model-based forecasting);
 - **comparison...hyperparameters_1.csv** — metrics with expanded hyperparameter set for the fund pool with loosened selection thresholds (see above);
 - **comparison...ver_6.csv** — metrics on correctly aggregated monthly data;
 - **comparison...ver_7.csv** — metrics on correctly aggregated monthly data with expanded hyperparameter set.
 - **comparison...ver_10.csv** — differs from ver_7 only by added data — January 2026, i.e., +1 month of training and test shifted to December–January 2026
