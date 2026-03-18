[🇷🇺 Русская версия](README_RU.md)

# Payment Data Analysis and Forecasting

Analyzing donor payment data to nonprofit funds, forecasting payment amounts in various forms — individual payments based on descriptive features, total payment sums for the next week, next month, and several time periods ahead, as well as grouped by various descriptive features (funds, payment providers, subscription type, payment categories).

In **first-try-origin.ipynb** — data is prepared for amount forecasting based on descriptive features without time component or feature grouping; a single model trained on all data that predicts the payment amount. This file also includes data analysis (work in progress).

In **first-try-sarimax.ipynb** — data is prepared for next-day amount forecasting using SARIMAX (for time series) as an intermediate step, to be used for generating an additional feature when training a CatBoost-based model.

In **first-try-cb_sarimax.ipynb** — data is prepared for amount forecasting based on descriptive features with time component (SARIMAX forecast as an additional feature) and feature grouping.

In **first-try-3-models.ipynb** — data is prepared for amount forecasting based on descriptive features with time component using 3 models (LSTM, SARIMAX, CatBoost) and weighted prediction averaging using multiple methods.

This section is a work in progress.

**Data description**:

fund_id: fund ID
full_sum: payment amount
date: date in unixtimestamp
provider: ID of the payment provider
signup: subscription type: 0 — one-time payment, 1 — recurring payment
category_1, category_2, category_3, category_4, category_5, category_6, category_7: 0 — does NOT belong to this category, 1 — belongs to this category
