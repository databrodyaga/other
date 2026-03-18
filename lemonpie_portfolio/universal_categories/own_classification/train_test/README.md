[🇷🇺 Русская версия](README_RU.md)

# ML Model Selection, Training, and Testing

This folder contains training pipelines for data preprocessing, feature generation, model and parameter selection, and testing combinations of all the above.

Details:
- classify_model_choice_*.ipynb — model and parameter selection on different datasets,
  - in version 1 — text fields after embedding generation were encoded as: purpose: 3PCA + mean; articles, projects, counterparties: mean (only),
  - in version 2 — all 4 text fields (purpose, articles, projects, counterparties) after embedding generation were encoded as 3PCA + mean,
- classify_forecast_metrics_.ipynb — metrics calculation as of a given date,
- model_re_train — pipeline for model retraining and labeling dictionary updates,

Tests:
- tests_1 — training baseline and advanced models on feature preparation variant 1 with synthetic tests: 900 examples (100 per class) from the original dataset, partially corrected texts, and heavily distorted texts;
- tests_2 — same model pipeline as test 1, with prediction testing on new data obtained after training and labeled with ground truth from the old dictionary;
- test_3 — similar to test 2, but conducted after a massive prediction upload past the training date; metrics are also high on payments with "old" articles;
- test_4 — metric testing on payments with articles the model had never seen (see drawbacks of the previous two tests). After the mass upload, we took 1721 payments where articles differ from the training dictionary (main_dict), labeled new articles in a new staging dictionary, and evaluated universal category labels from previously uploaded model predictions against ground truth (expert-labeled). Achieved average accuracy of 71% on new data;
- test_5 — switched data preprocessing to scheme 2 with more detailed text encoding (see above), and ran standard tests from example 1 (synthetically assembled datasets from real data); metrics improved relative to test 1;
- test_6 — ran the test 5 model on real new data from test 4 (same 1721 examples); average accuracy 80%, but the test dataset is highly imbalanced — rare classes are poorly predicted. Keeping the model in production (retraining on the updated article dictionary), collecting statistics until the next retraining.
