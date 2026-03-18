[🇷🇺 Русская версия](README_RU.md)

# Model Training and Testing

In this section we reworked the original data preprocessing and model training script from second_try (with daily aggregation) for monthly aggregated data. In tests, the 3-model ensemble proved less viable due to the critical reduction in data volume — time series of several hundred or thousand days were reduced 30x to just a few dozen data points.

Dataset — all payments up to 01.01.2026; the same set of funds was used across all tests (preprocessing is identical in all scripts).

**Original ensembles (ish / v3 files)**:

ish_v1 (1-month test):
	•	median MASE ≈ 1.23
	•	mean MASE ≈ 2.78
	•	high instability
	•	no clear advantage over CatBoost

ish_2mes_test (2-month test):
	•	median MASE ≈ 1.36
	•	degradation with increased test horizon
	•	ridge ensemble often worse than nn ensemble

v3:
	•	median MASE ≈ 1.51
	•	worst of the ensembles
	•	essentially a regression


**Individual SARIMAX and CatBoost (vanilla) models**:

vanilla_v2 / vanilla_2mes_test (identical results)

sarimax:
	•	median MASE ≈ 1.07
	•	highly unstable: excellent in some cases, MASE 3–7 in others.

catboost:
	•	median MASE ≈ 0.65
	•	mean MASE ≈ 1.48 (due to heavy tails)
	•	best single-model result across all experiments

Vanilla conclusion: CatBoost is significantly better than SARIMAX, consistently wins on most funds, holds up well even on the 2-month test.


**Overall metrics summary**: ensembles do NOT provide a stable advantage over standalone CatBoost (worse median, heavier tails, higher sensitivity to test length); standalone CatBoost without ensembling is the optimal choice.

Files in this folder:
 - **comparison...v1/v3/v4.csv** — metrics for the original model set with different hyperparameters, tested on the last one or two months;
 - **comparison...v1/v3/v4.csv** — metrics for individual SARIMAX and CatBoost models tested on the last one or two months;
 - **third_try_vanil_models.py, third_try.py** — data preprocessing and training scripts for ensembles and individual models.
