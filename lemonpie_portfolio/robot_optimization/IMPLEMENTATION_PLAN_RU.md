# Реализация без SQL: служебные таблицы + классификация

## Что в итоге
Сделано 2 независимых скрипта:
1. `scripts/build_service_tables.py` — генерирует 2 служебные таблицы из исходников.
2. `scripts/classify_incoming_payments.py` — классифицирует новые входящие платежи по статьям, используя эти таблицы.

## Служебные таблицы
### Таблица 1 (ключевики)
Поля:
- `client_id`
- `keywords_raw`
- `keywords_norm`
- `keywords_logic`
- `article_name`
- `keyword_vector`
- `article_vector`
- `counterparty`

### Таблица 2 (история назначений)
Поля:
- `client_id`
- `payment_id`
- `payment_date`
- `counterparty`
- `purpose_raw`
- `purpose_norm`
- `article_name`
- `purpose_vector`
- `article_vector`

Вектора сохраняются как JSON-строки.

## Нормализация и вектора
- Нормализация назначений: regex + стоп-слова (из вашего блока).
- Лемматизация: `ru_core_news_sm`.
- Вектора: `doc.vector` из `ru_core_news_sm` (для текста ключевиков, назначений и названий статей).

Установка модели реализована строго по вашему шаблону в обоих скриптах:
- сначала `pip install ru-core-news-sm==3.6.0`
- fallback на установку wheel из `MODEL_WHL`.

## Логика классификации (Этапы 1/2/3)
1. Этап 1:
- свои правила (ключевики + AND/OR)
- своя история (точное совпадение `purpose_norm` или `counterparty`)
- чужие правила (как кандидаты)

2. Этап 2:
- векторный top-k поиск по:
  - таблице нормализованных назначений (`purpose_vector`)
  - таблице ключевиков (`keyword_vector`)

3. Этап 3:
- приоритет «своих» кандидатов
- иначе маппинг чужой статьи на свою статью клиента по близости векторов названий статей
- иначе majority vote, пометка `new_article_candidate`

## Запуск
1. Сгенерировать служебные таблицы:
```bash
python3 scripts/build_service_tables.py \
  --rules-source robots.csv \
  --payments-source data_full.parquet \
  --out-keywords artifacts/service_keywords.parquet \
  --out-history artifacts/service_history.parquet
```

2. Классифицировать новые платежи (входной CSV аналогичен структуре `data_full.parquet`):
```bash
python3 scripts/classify_incoming_payments.py \
  --service-keywords artifacts/service_keywords.parquet \
  --service-history artifacts/service_history.parquet \
  --incoming-csv incoming_new_payments.csv \
  --incoming-sep ',' \
  --out artifacts/classified_incoming.csv
```

## Маппинг `email польз` -> `accounts__user_id`
Добавлена опция `--client-map-source` для явного маппинга в единый `client_id`, если нужно привести правила и историю к одному идентификатору.
