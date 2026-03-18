[🇷🇺 Русская версия](README_RU.md)

# Feature Selection

In this section we select features based on LSTM (Occlusion Sensitivity) and CatBoost (feature selection) models; attached files contain metric results for the former and importance scores for the latter.

**Features that can be excluded universally (across all funds)**:

The following features demonstrate low importance in virtually all funds. The share of funds with importance < 1.0 exceeds 80–90% for each, in some cases 100% — these features are barely used by the models and contribute minimally to predictions:

day_purpose_pca_4, day_purpose_pca_5
In most funds they show near-zero metrics or very low importance, unlike the leading components.

day_robots__id_nunique
day_of_week_sin, day_of_week_cos, week_number_sin, week_number_cos — subject to additional testing
article_id_ufreq_std
projects__parent_id_ufreq_std
counterpartie_id_ufreq_std
day_counterpartie_id_nunique
day_projects__parent_id_nunique
day_donor_id_nunique
donor_id_ufreq_std
account_id_ufreq_std
day_donor_cat_id_nunique
day_articles__parent_id_nunique
day_counterparties__parent_id_nunique
donor_id_ufreq_mean
day_account_id_nunique
counterpartie_id_ufreq_mean
counterparties__parent_id_ufreq_std
day_article_id_nunique
project_id_ufreq_std
projects__parent_id_ufreq_mean
day_project_id_nunique
articles__parent_id_ufreq_std
account_id_ufreq_mean


**Comparison of day_purpose_mean vs 5 PCA components**:
*CatBoost*: PCA components (especially the first 1–3) often show higher individual and aggregate importance than day_purpose_mean.
*LSTM*:
day_purpose_mean: impact is unstable — sometimes useful, sometimes its removal improves the model.
PCA components: Also unstable individually. The first components (pca_1, pca_2, pca_3) more often prove important (their removal worsens the metric for some funds), while pca_4, pca_5 can more often be removed without harm or even with benefit.
**Conclusion**:
A set of several PCA components (e.g., the first 3) is likely more informative than a single day_purpose_mean. day_purpose_mean can be kept as a supplement — it does not strongly correlate with PCA components and adds its own unique (albeit small) importance.

Based on the above conclusions, a pool of features for exclusion was formed — feature_set_2 in the comparison file. Additionally, other feature sets were tested with metrics calculated on the test set across the same set of funds.
Data is provided in the **comparison** file on tabs with corresponding labels:
- excluded feature sets are specified,
- absolute metric values,
- normalized (to the minimum value of one of the models for a given fund) RMSE metric values,
to evaluate individual models and ensembles against each other; averaged values across all funds allow comparison of prediction results across different feature sets (between tabs and the summary on the comparison tab).

Similar data is also provided for the most effective feature sets for the top 10 funds by data volume (last tab — comparison_normalized).

**In summary**, feature_set_2 based on importance filtering and sequential feature zeroing showed one of the best results; a slightly refined version — **feature_set_6** — was ultimately selected for production use (slightly fewer features, whose exclusion simplifies the training algorithm + one feature was restored that showed consistent metric improvement).
**Across different models and ensembles**, occasional CatBoost efficiency spikes on individual funds on average lose to the 3-model ensemble with weights optimized by a simple neural network (3models_ens_nn_rmse).

Files in this folder:
 - **second_try_3_models_features_choice.py** — training and prediction script
 - **request_3.sql** — database query for data export
 - **users_actual.csv** — active user_id (funds) for raw data filtering
 - **2try_data_with_purpose_mean.parquet** — cache files with language model embedding calculations for text field encoding and correlation tables
 - **catboost_feature_importances.csv** — feature importance from the CatBoost model (all features)
 - **feature_importance_lstm_.csv** — LSTM metrics when zeroing out features one by one
 - **comparison.xlsx** — summary table with metrics for different datasets and models
