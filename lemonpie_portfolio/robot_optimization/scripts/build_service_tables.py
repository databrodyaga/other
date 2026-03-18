#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

MODEL_WHL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "ru_core_news_sm-3.6.0/ru_core_news_sm-3.6.0-py3-none-any.whl"
)

STOP_WORDS = {
    "оплата", "платеж", "платёж", "по", "за", "от", "с", "на", "в", "без", "ндс"
}


def normalize_id(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""
    if re.fullmatch(r"[+-]?\d+\.0+", s):
        return s.split(".", 1)[0]
    return s


def normalize_direction(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    s = str(value).strip().lower()
    if s in {"", "nan", "none", "null"}:
        return ""
    return s


def is_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def ensure_spacy_model() -> None:
    if not is_package("spacy"):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "spacy>=3.6,<3.8"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    if not is_package("ru_core_news_sm"):
        try:
            # для локальной установки
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "ru-core-news-sm==3.6.0"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            # для запуске в jobs
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    MODEL_WHL,
                ],
                stdout=subprocess.DEVNULL,
            )


def ensure_pandas():
    try:
        import pandas as pd  # type: ignore

        return pd
    except Exception as exc:
        raise RuntimeError("Требуется pandas/pyarrow: pip install pandas pyarrow") from exc


def load_df(path: Path, csv_sep: Optional[str] = None):
    pd = ensure_pandas()
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        if csv_sep:
            return pd.read_csv(path, sep=csv_sep)
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, sep=";")
    raise ValueError(f"Неподдерживаемый формат: {path}")


def save_df(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Неподдерживаемый формат вывода: {path}")


def normalize_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"\\b\\d{1,2}[./]\\d{1,2}[./]\\d{2,4}\\b", " ", text)
    text = re.sub(r"\\b\\d{4,}\\b", " ", text)
    text = re.sub(r"\\b\\d+[.,]\\d+\\b", " ", text)
    text = re.sub(r"[^a-zа-яё\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()

    tokens = [t for t in text.split() if t not in STOP_WORDS]
    return " ".join(tokens)


def _iter_docs_with_progress(texts: List[str], nlp, desc: str):
    docs = nlp.pipe(texts, batch_size=1024)
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(docs, total=len(texts), desc=desc)
    except Exception:
        return docs


def _iter_items_with_progress(items: List[str], desc: str):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(items, total=len(items), desc=desc)
    except Exception:
        return items


def lemmatize_series(texts: Iterable[str], nlp, desc: str) -> List[str]:
    seq = list(texts)
    docs = _iter_docs_with_progress(seq, nlp, desc)
    return [" ".join([t.lemma_ for t in doc if t.is_alpha]) for doc in docs]


def vectorize_texts(texts: Iterable[str], nlp, desc: str) -> List[str]:
    seq = list(texts)
    docs = _iter_docs_with_progress(seq, nlp, desc)
    return [json.dumps(doc.vector.tolist(), ensure_ascii=False) for doc in docs]


def _unique_preserve_order(texts: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for t in texts:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def lemmatize_unique_map(texts: Iterable[str], nlp, desc: str) -> Dict[str, str]:
    unique_texts = _unique_preserve_order(texts)
    lemmas = lemmatize_series(unique_texts, nlp, f"{desc} (unique={len(unique_texts)})")
    return dict(zip(unique_texts, lemmas))


def vectorize_unique_map(texts: Iterable[str], nlp, desc: str) -> Dict[str, str]:
    unique_texts = _unique_preserve_order(texts)
    vectors = vectorize_texts(unique_texts, nlp, f"{desc} (unique={len(unique_texts)})")
    return dict(zip(unique_texts, vectors))


def normalize_unique_map(texts: Iterable[str], desc: str) -> Dict[str, str]:
    unique_texts = _unique_preserve_order(texts)
    normalized = [normalize_text(t) for t in _iter_items_with_progress(unique_texts, f"{desc} (unique={len(unique_texts)})")]
    return dict(zip(unique_texts, normalized))


def maybe_apply_client_map(
    series,
    client_map_df,
    map_from_col: Optional[str],
    map_to_col: Optional[str],
):
    if client_map_df is None or not map_from_col or not map_to_col:
        return series.astype(str)

    mapping: Dict[str, str] = (
        client_map_df[[map_from_col, map_to_col]]
        .dropna()
        .astype(str)
        .drop_duplicates(subset=[map_from_col])
        .set_index(map_from_col)[map_to_col]
        .to_dict()
    )
    return series.astype(str).map(lambda x: mapping.get(x, x))


def build_keywords_table(args) -> Any:
    pd = ensure_pandas()
    rules_df = load_df(Path(args.rules_source), csv_sep=args.rules_sep)

    if args.rules_status_col in rules_df.columns and args.rules_enabled_value:
        status_expected = str(args.rules_enabled_value).strip()
        rules_df = rules_df[
            rules_df[args.rules_status_col].astype(str).str.strip() == status_expected
        ]
    if args.rules_type_col in rules_df.columns and args.rules_incoming_value:
        type_expected = str(args.rules_incoming_value).strip().lower()
        rules_df = rules_df[
            rules_df[args.rules_type_col].astype(str).str.strip().str.lower() == type_expected
        ]

    client_map_df = load_df(Path(args.client_map_source), csv_sep=args.client_map_sep) if args.client_map_source else None

    out = rules_df.copy()
    out["client_id"] = maybe_apply_client_map(
        out[args.rules_client_col],
        client_map_df,
        args.client_map_rules_col,
        args.client_map_target_col,
    )
    # Приводим user_id правил к float, чтобы тип совпадал с accounts__user_id из платежей.
    out["client_id"] = pd.to_numeric(out["client_id"], errors="coerce").astype("float64")

    out["keywords_raw"] = out[args.rules_keywords_col].fillna("").astype(str)
    out["keywords_logic"] = out[args.rules_logic_col].fillna("or").astype(str).str.lower()
    out["direction"] = (
        out[args.rules_type_col].map(normalize_direction)
        if args.rules_type_col in out.columns
        else ""
    )
    out["article_name"] = out[args.rules_article_col].fillna("").astype(str)
    out["counterparty"] = out[args.rules_counterparty_col].fillna("").astype(str) if args.rules_counterparty_col in out.columns else ""
    out["contractor_id"] = (
        out[args.rules_contractor_id_col].map(normalize_id)
        if args.rules_contractor_id_col in out.columns
        else ""
    )

    keywords_norm_map = normalize_unique_map(
        out["keywords_raw"].tolist(),
        desc="keywords -> normalize",
    )
    out["keywords_norm"] = out["keywords_raw"].map(keywords_norm_map)

    ensure_spacy_model()
    import spacy  # type: ignore

    try:
        nlp = spacy.load("ru_core_news_sm", disable=["parser", "ner"])
    except Exception:
        nlp = spacy.load("ru_core_news_sm")

    keyword_vec_map = vectorize_unique_map(
        out["keywords_norm"].tolist(),
        nlp,
        desc="keywords -> vectors",
    )
    article_vec_map = vectorize_unique_map(
        out["article_name"].tolist(),
        nlp,
        desc="keyword articles -> vectors",
    )
    out["keyword_vector"] = out["keywords_norm"].map(keyword_vec_map)
    out["article_vector"] = out["article_name"].map(article_vec_map)

    out = out[
        [
            "client_id",
            "keywords_raw",
            "keywords_norm",
            "keywords_logic",
            "direction",
            "article_name",
            "keyword_vector",
            "article_vector",
            "counterparty",
            "contractor_id",
        ]
    ].copy()
    return out


def build_history_table(args) -> Any:
    pd = ensure_pandas()
    payments_df = load_df(Path(args.payments_source), csv_sep=args.payments_sep)

    client_map_df = load_df(Path(args.client_map_source), csv_sep=args.client_map_sep) if args.client_map_source else None

    out = payments_df.copy()
    out["client_id"] = maybe_apply_client_map(
        out[args.payments_client_col],
        client_map_df,
        args.client_map_payments_col,
        args.client_map_target_col,
    )

    out["payment_id"] = out[args.payments_id_col].fillna("").astype(str) if args.payments_id_col in out.columns else ""
    out["payment_date"] = out[args.payments_date_col].fillna("").astype(str) if args.payments_date_col in out.columns else ""
    out["counterparty"] = out[args.payments_counterparty_col].fillna("").astype(str) if args.payments_counterparty_col in out.columns else ""
    out["direction"] = (
        out[args.payments_direction_col].map(normalize_direction)
        if args.payments_direction_col in out.columns
        else ""
    )

    out["purpose_raw"] = out[args.payments_purpose_col].fillna("").astype(str)
    out["article_name"] = out[args.payments_article_col].fillna("").astype(str)

    purpose_clean_map = normalize_unique_map(
        out["purpose_raw"].tolist(),
        desc="purposes -> normalize",
    )
    out["purpose_clean"] = out["purpose_raw"].map(purpose_clean_map)

    ensure_spacy_model()
    import spacy  # type: ignore

    try:
        nlp = spacy.load("ru_core_news_sm", disable=["parser", "ner"])
    except Exception:
        nlp = spacy.load("ru_core_news_sm")

    purpose_norm_map = lemmatize_unique_map(
        out["purpose_clean"].tolist(),
        nlp,
        desc="purposes -> lemmas",
    )
    out["purpose_norm"] = out["purpose_clean"].map(purpose_norm_map)

    out = out[
        [
            "client_id",
            "payment_id",
            "payment_date",
            "purpose_norm",
            "direction",
            "article_name",
        ]
    ].copy()
    if args.history_drop_empty_purpose:
        out = out[out["purpose_norm"].str.strip() != ""]
    if args.history_drop_empty_article:
        out = out[out["article_name"].str.strip() != ""]

    # Для каждого (client_id, purpose_norm, direction) оставляем последнюю запись:
    # сначала по payment_date, затем по payment_id.
    out["_payment_date_dt"] = pd.to_datetime(out["payment_date"], errors="coerce")
    out["_payment_id_num"] = pd.to_numeric(out["payment_id"], errors="coerce")
    out["_row_num"] = range(len(out))
    out = out.sort_values(
        by=["client_id", "purpose_norm", "direction", "_payment_date_dt", "_payment_id_num", "_row_num"],
        ascending=[True, True, True, True, True, True],
        na_position="first",
    )
    out = out.drop_duplicates(
        subset=["client_id", "purpose_norm", "direction"],
        keep="last",
    ).copy()
    out = out.drop(columns=["_payment_date_dt", "_payment_id_num", "_row_num"])

    purpose_vec_map = vectorize_unique_map(
        out["purpose_norm"].tolist(),
        nlp,
        desc="purposes -> vectors",
    )
    article_vec_map = vectorize_unique_map(
        out["article_name"].tolist(),
        nlp,
        desc="history articles -> vectors",
    )
    out["purpose_vector"] = out["purpose_norm"].map(purpose_vec_map)
    out["article_vector"] = out["article_name"].map(article_vec_map)

    out = out.rename(
        columns={
            "payment_id": "last_payment_id",
            "payment_date": "last_payment_date",
        }
    )

    out = out[
        [
            "client_id",
            "last_payment_id",
            "last_payment_date",
            "purpose_norm",
            "direction",
            "purpose_vector",
            "article_name",
            "article_vector",
        ]
    ].copy()
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Генерация служебных таблиц для классификации платежей")

    p.add_argument("--rules-source", required=True, help="Источник правил (robots.csv)")
    p.add_argument("--payments-source", required=True, help="Источник истории платежей (csv/parquet)")
    p.add_argument("--out-keywords", required=True, help="Выходной файл таблицы ключевиков (csv/parquet)")
    p.add_argument("--out-history", required=True, help="Выходной файл таблицы истории (csv/parquet)")

    p.add_argument("--rules-sep", default=",", help="Разделитель CSV для правил")
    p.add_argument("--payments-sep", default=None, help="Разделитель CSV для платежей")

    p.add_argument("--rules-client-col", default="user_id")
    p.add_argument("--rules-keywords-col", default="text")
    p.add_argument("--rules-logic-col", default="mode")
    p.add_argument("--rules-article-col", default="article_name")
    p.add_argument("--rules-counterparty-col", default="counterpartie_name")
    p.add_argument("--rules-contractor-id-col", default="contractor_id")
    p.add_argument("--rules-status-col", default="status")
    p.add_argument("--rules-type-col", default="expenditure")
    p.add_argument("--rules-enabled-value", default="1")
    p.add_argument("--rules-incoming-value", default="")

    p.add_argument("--payments-client-col", default="accounts__user_id")
    p.add_argument("--payments-id-col", default="id")
    p.add_argument("--payments-date-col", default="date")
    p.add_argument("--payments-counterparty-col", default="counterparties__name")
    p.add_argument("--payments-direction-col", default="expenditure")
    p.add_argument("--payments-contractor-id-col", default="contractor_id")
    p.add_argument("--payments-purpose-col", default="purpose")
    p.add_argument("--payments-article-col", default="articles__name")

    p.add_argument("--client-map-source", default=None, help="Опциональный файл маппинга client_id")
    p.add_argument("--client-map-sep", default=",", help="Разделитель CSV маппинга")
    p.add_argument("--client-map-rules-col", default="user_id", help="Колонка id правил в маппинге")
    p.add_argument("--client-map-payments-col", default="accounts__user_id", help="Колонка accounts__user_id в маппинге")
    p.add_argument("--client-map-target-col", default="client_id", help="Единая колонка-цель в маппинге")
    p.add_argument(
        "--history-drop-empty-purpose",
        action="store_true",
        help="Исключать строки с пустым purpose_norm из service_history",
    )
    p.add_argument(
        "--history-drop-empty-article",
        action="store_true",
        help="Исключать строки с пустым article_name из service_history",
    )

    return p


def main() -> None:
    args = build_parser().parse_args()

    kw = build_keywords_table(args)
    hist = build_history_table(args)

    save_df(kw, Path(args.out_keywords))
    save_df(hist, Path(args.out_history))

    print(
        json.dumps(
            {
                "status": "ok",
                "keywords_rows": int(len(kw)),
                "history_rows": int(len(hist)),
                "out_keywords": args.out_keywords,
                "out_history": args.out_history,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
