[🇷🇺 Русская версия](README_RU.md)

# Payment Classification by Budget Articles

## What is This
The project classifies new payments by budget article for each client/fund (`accounts__user_id`), using:
- keyword-based rules (`robots_export_old.csv`);
- payment history (`data_full_inc_out.parquet`);
- text normalization + lemmatization (`ru_core_news_sm`);
- vector search over normalized payment purposes and keywords.

## What Was Done
- Implemented service table generator:
  - `artifacts/service_keywords.parquet`
  - `artifacts/service_history.parquet`
- Implemented new payment classifier:
  - `scripts/classify_payments.py`
- Added payment direction filtering (`incoming`/`outgoing`) in service tables and classification.
- Added `contractor_id` support for precise stages.

## Data and Testing
- Test sample: `24,837` payments (incoming + outgoing), February 2026.
- Rules: up to mid-November 2025.
- Service history: up to February 2026.
- Target payments were not used as answer sources during testing.

## Overall Metrics
- `Coverage`: `1.0000`
- `Accuracy`: `0.8368`
- `F1 macro`: `0.5737`
- `F1 weighted`: `0.8346`

## Metrics by Stage
| decision_stage | rows | accuracy | comment |
|---|---:|---:|---|
| stage1_own_rules_own_priority | 12343 | 0.908936 | Stage 1: own keywords matched (final priority) |
| stage1_own_history_own_priority | 8862 | 0.928459 | Stage 1: exact/close match from own payment history |
| stage2_history_own_priority | 1423 | 0.674631 | Stage 2: vector search over normalized payments across all clients, own result won |
| stage2_keywords_own_priority | 0 | 0.000000 | Stage 2: vector search over keywords across all clients, own result won |
| stage3_foreign_to_own_mapping | 123 | 0.325203 | Stage 3: foreign category mapped to closest own category |
| stage3_majority_vote | 2086 | 0.161553 | Stage 3: fallback, voting among foreign category candidates |
| no_candidates | 0 | 0.000000 | No candidates found |
| stage3_failed | 0 | 0.000000 | Final selection failed |

Traffic share by stage:
- `stage1_rules`: `49.7%`
- `stage1_history`: `35.7%`
- `stage2_history`: `5.7%`
- `stage3_mapping`: `0.5%`
- `stage3_majority`: `8.4%`

## Metrics by suggestion_type
| suggestion_type | rows | accuracy | includes |
|---|---:|---:|---|
| existing_article | 22751 | 0.898730 | `stage1_own_rules_own_priority`, `stage1_own_history_own_priority`, `stage2_*_own_priority`, `stage3_foreign_to_own_mapping` |
| new_article_candidate | 2086 | 0.161553 | `stage3_majority_vote` |

## Business Interpretation
- Using only keyword robots (equivalent to `stage1_own_rules_own_priority`) would yield ~`50%` coverage.
- Adding own history provides another `8,862` payments at ~`92.8%` accuracy.
- Stage 2 adds coverage but with lower quality (`~67%`).
- Stage 3 is a fallback for new/complex cases; requires active review.
- Practical mode: auto-approve `stage1` and `existing_article`; send `stage3_majority_vote` for confirmation.

## Important Notes on Metrics
- Metrics are calculated by **exact text match** of the article name.
- Formally different but semantically close names count as errors:
  - `ФОТ` vs `ФОТ - зп`
  - `Страховые взносы` vs `Cтраховой взнос`

## Stage Logic and Thresholds
1. `stage1_own_rules_own_priority`: own keywords (`score=1.00` with `contractor_id`, `0.98` without).
2. `stage1_own_history_own_priority`: exact `purpose_norm` match (`score=0.97` with `contractor_id`, `0.95` without).
3. `stage1_foreign_rules` (intermediate): foreign keywords (`score=0.75`) as candidates.
4. Stage 2 is triggered if no candidates or `max_score < vector_threshold` (`0.82`):
   - search over `history purpose_vector`;
   - search over `keywords keyword_vector`.
5. If own candidates exist among results, final decision is `*_own_priority`.
6. If no own candidates:
   - `stage3_foreign_to_own_mapping` when similarity `>= own_similarity_threshold` (`0.80`);
   - otherwise `stage3_majority_vote`.

## Running
### 1) Rebuilding service tables
```bash
python scripts/build_service_tables.py \
  --rules-source robots_export_old.csv \
  --payments-source data_full_inc_out.parquet \
  --out-keywords artifacts/service_keywords.parquet \
  --out-history artifacts/service_history.parquet
```

### 2) Classifying new payments
```bash
python scripts/classify_payments.py \
  --service-keywords artifacts/service_keywords.parquet \
  --service-history artifacts/service_history.parquet \
  --incoming-csv incoming_new_payments.csv \
  --out artifacts/classified_incoming.csv
```

## Future Improvements
- Add `project` and `donor` prediction (following the same approach as `article`).
- Introduce a canonicalization dictionary for Latin/transliteration (`Cloudpayments`, `Cloud payments` -> single canonical form).
- Optimize generation and classification scripts (redundant checks and duplicate passes).
