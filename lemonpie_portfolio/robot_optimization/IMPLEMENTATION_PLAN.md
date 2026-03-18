[🇷🇺 Русская версия](IMPLEMENTATION_PLAN_RU.md)

# SQL-Free Implementation: Service Tables + Classification

## Summary
Two independent scripts were built:
1. `scripts/build_service_tables.py` — generates 2 service tables from source data.
2. `scripts/classify_incoming_payments.py` — classifies new incoming payments by budget articles using these tables.

## Service Tables
### Table 1 (Keywords)
Fields:
- `client_id`
- `keywords_raw`
- `keywords_norm`
- `keywords_logic`
- `article_name`
- `keyword_vector`
- `article_vector`
- `counterparty`

### Table 2 (Payment Purpose History)
Fields:
- `client_id`
- `payment_id`
- `payment_date`
- `counterparty`
- `purpose_raw`
- `purpose_norm`
- `article_name`
- `purpose_vector`
- `article_vector`

Vectors are stored as JSON strings.

## Normalization and Vectors
- Purpose normalization: regex + stop words.
- Lemmatization: `ru_core_news_sm`.
- Vectors: `doc.vector` from `ru_core_news_sm` (for keyword text, payment purposes, and article names).

Model installation follows the established pattern in both scripts:
- first `pip install ru-core-news-sm==3.6.0`
- fallback to wheel installation from `MODEL_WHL`.

## Classification Logic (Stages 1/2/3)
1. Stage 1:
- own rules (keywords + AND/OR)
- own history (exact `purpose_norm` or `counterparty` match)
- foreign rules (as candidates)

2. Stage 2:
- vector top-k search over:
  - normalized payment purpose table (`purpose_vector`)
  - keyword table (`keyword_vector`)

3. Stage 3:
- own candidates take priority
- otherwise map a foreign article to the client's own article by article name vector similarity
- otherwise majority vote, labeled as `new_article_candidate`

## Running
1. Generate service tables:
```bash
python3 scripts/build_service_tables.py \
  --rules-source robots.csv \
  --payments-source data_full.parquet \
  --out-keywords artifacts/service_keywords.parquet \
  --out-history artifacts/service_history.parquet
```

2. Classify new payments (input CSV matches `data_full.parquet` structure):
```bash
python3 scripts/classify_incoming_payments.py \
  --service-keywords artifacts/service_keywords.parquet \
  --service-history artifacts/service_history.parquet \
  --incoming-csv incoming_new_payments.csv \
  --incoming-sep ',' \
  --out artifacts/classified_incoming.csv
```

## Email -> `accounts__user_id` Mapping
Added `--client-map-source` option for explicit mapping to a unified `client_id`, if rules and history need to be aligned to a single identifier.
