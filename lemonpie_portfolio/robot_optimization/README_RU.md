# Классификация платежей по статьям бюджета

## Что это
Проект классифицирует новые платежи по статье бюджета для каждого клиента/фонда (`accounts__user_id`), используя:
- правила с ключевыми словами (`robots_export_old.csv`);
- историю платежей (`data_full_inc_out.parquet`);
- нормализацию текста + лемматизацию (`ru_core_news_sm`);
- векторный поиск по нормализованным назначениям и ключевикам.

## Что сделано
- Реализован генератор сервисных таблиц:
  - `artifacts/service_keywords.parquet`
  - `artifacts/service_history.parquet`
- Реализован классификатор новых платежей:
  - `scripts/classify_payments.py`
- Добавлена фильтрация по направлению платежа (`incoming`/`outgoing`) в сервисных таблицах и в классификации.
- Добавлен учет `contractor_id` для точных стадий.

## Данные и тест
- Тестовая выборка: `24,837` платежей (входящие + исходящие), февраль 2026.
- Правила: до середины ноября 2025.
- История для сервиса: до февраля 2026.
- В тесте целевые платежи не использовались как источник ответов.

## Общие метрики
- `Coverage`: `1.0000`
- `Accuracy`: `0.8368`
- `F1 macro`: `0.5737`
- `F1 weighted`: `0.8346`

## Метрики по стадиям
| decision_stage | rows | accuracy | comment |
|---|---:|---:|---|
| stage1_own_rules_own_priority | 12343 | 0.908936 | Этап 1: сработали свои ключевые слова (финальный приоритет) |
| stage1_own_history_own_priority | 8862 | 0.928459 | Этап 1: сработало точное/близкое совпадение по своей истории платежей |
| stage2_history_own_priority | 1423 | 0.674631 | Этап 2: векторный поиск по нормализованным платежам всей базы, победил свой результат |
| stage2_keywords_own_priority | 0 | 0.000000 | Этап 2: векторный поиск по ключевикам всей базы, победил свой результат |
| stage3_foreign_to_own_mapping | 123 | 0.325203 | Этап 3: чужая категория сопоставлена с ближайшей своей категорией |
| stage3_majority_vote | 2086 | 0.161553 | Этап 3: запасной вариант, голосование по кандидатам из чужих категорий |
| no_candidates | 0 | 0.000000 | Кандидаты не найдены |
| stage3_failed | 0 | 0.000000 | Финальный выбор не выполнен |

Доли трафика по стадиям:
- `stage1_rules`: `49.7%`
- `stage1_history`: `35.7%`
- `stage2_history`: `5.7%`
- `stage3_mapping`: `0.5%`
- `stage3_majority`: `8.4%`

## Метрики по suggestion_type
| suggestion_type | rows | accuracy | что входит |
|---|---:|---:|---|
| existing_article | 22751 | 0.898730 | `stage1_own_rules_own_priority`, `stage1_own_history_own_priority`, `stage2_*_own_priority`, `stage3_foreign_to_own_mapping` |
| new_article_candidate | 2086 | 0.161553 | `stage3_majority_vote` |

## Бизнес-интерпретация
- Если использовать только роботов-ключевики (аналог `stage1_own_rules_own_priority`), покрытие было бы около `50%`.
- Добавление своей истории дает еще `8,862` платежей с точностью около `92.8%`.
- Stage 2 добавляет покрытие, но с более низким качеством (`~67%`).
- Stage 3 — fallback для новых/сложных случаев; требует активной проверки.
- Практический режим: автоматом отдавать `stage1` и `existing_article`, `stage3_majority_vote` — на подтверждение.

## Важное про метрики
- Метрики считаются по **точному совпадению текста** статьи.
- Формально разные, но близкие названия считаются ошибкой:
  - `ФОТ` vs `ФОТ - зп`
  - `Страховые взносы` vs `Cтраховой взнос`

## Логика стадий и пороги
1. `stage1_own_rules_own_priority`: свои ключевые слова (`score=1.00` с `contractor_id`, `0.98` без).
2. `stage1_own_history_own_priority`: точное совпадение `purpose_norm` (`score=0.97` с `contractor_id`, `0.95` без).
3. `stage1_foreign_rules` (промежуточно): чужие ключевики (`score=0.75`) как кандидаты.
4. Stage 2 запускается если нет кандидатов или `max_score < vector_threshold` (`0.82`):
   - поиск по `history purpose_vector`;
   - поиск по `keywords keyword_vector`.
5. Если среди кандидатов есть свой, финал сразу `*_own_priority`.
6. Если своих нет:
   - `stage3_foreign_to_own_mapping` при близости `>= own_similarity_threshold` (`0.80`);
   - иначе `stage3_majority_vote`.

## Запуск
### 1) Пересборка сервисных таблиц
```bash
python scripts/build_service_tables.py \
  --rules-source robots_export_old.csv \
  --payments-source data_full_inc_out.parquet \
  --out-keywords artifacts/service_keywords.parquet \
  --out-history artifacts/service_history.parquet
```

### 2) Классификация новых платежей
```bash
python scripts/classify_payments.py \
  --service-keywords artifacts/service_keywords.parquet \
  --service-history artifacts/service_history.parquet \
  --incoming-csv incoming_new_payments.csv \
  --out artifacts/classified_incoming.csv
```

## Дальнейшие доработки
- Добавить прогноз `project` и `donor` (по схеме, аналогичной `article`).
- Ввести словарь канонизации латиницы/транслита (`Cloudpayments`, `Cloud payments` -> единый канон).
- Оптимизировать скрипты генерации и классификации (повторные проверки и дублирующие проходы).
