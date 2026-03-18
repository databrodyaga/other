#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

MODEL_WHL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "ru_core_news_sm-3.6.0/ru_core_news_sm-3.6.0-py3-none-any.whl"
)

STOP_WORDS = {
    "оплата", "платеж", "платёж", "по", "за", "от", "с", "на", "в", "без", "ндс"
}


@dataclass
class Candidate:
    article: str
    score: float
    source: str
    is_own: bool
    payload: Dict[str, Any]


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


def lemmatize_text(text: str, nlp) -> str:
    doc = nlp(text)
    return " ".join([t.lemma_ for t in doc if t.is_alpha])


def to_vec(doc) -> List[float]:
    return doc.vector.tolist()


def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return dot / (n1 * n2)


def parse_vec(raw: Any) -> List[float]:
    if not isinstance(raw, str) or not raw:
        return []
    try:
        return list(json.loads(raw))
    except Exception:
        return []


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


def split_keywords(raw_keywords: str) -> List[str]:
    if not raw_keywords:
        return []
    return [p.strip().lower() for p in raw_keywords.split(",") if p.strip()]


def keyword_match_details(text_raw: str, keywords_raw: str, logic: str) -> Tuple[bool, List[str]]:
    text = (text_raw or "").lower()
    kws = split_keywords(keywords_raw)
    if not kws:
        return (False, [])

    logic_norm = (logic or "or").strip().lower()
    hit_kws = [kw for kw in kws if kw in text]
    if logic_norm == "and":
        ok = len(hit_kws) == len(kws)
        return (ok, kws if ok else [])
    return (len(hit_kws) > 0, hit_kws)


def keyword_match(text_raw: str, keywords_raw: str, logic: str) -> bool:
    return keyword_match_details(text_raw, keywords_raw, logic)[0]


def _keyword_specificity(matched_keywords: List[str]) -> Tuple[int, int, int]:
    if not matched_keywords:
        return (0, 0, 0)
    return (len(matched_keywords), max(len(k) for k in matched_keywords), sum(len(k) for k in matched_keywords))


def stage1_own_rules(
    purpose_raw: str,
    contractor_id: str,
    rules_rows: List[Dict[str, Any]],
) -> Optional[Candidate]:
    scoped_matches: List[Tuple[Tuple[int, int, int], Dict[str, Any], List[str]]] = []
    generic_matches: List[Tuple[Tuple[int, int, int], Dict[str, Any], List[str]]] = []

    for r in rules_rows:
        matched, hit_kws = keyword_match_details(
            purpose_raw,
            str(r.get("keywords_raw", "")),
            str(r.get("keywords_logic", "or")),
        )
        if not matched:
            continue

        article = str(r.get("article_name", ""))
        if not article:
            continue

        rule_contractor_id = normalize_id(r.get("contractor_id", ""))
        item = (_keyword_specificity(hit_kws), r, hit_kws)
        if contractor_id and rule_contractor_id and contractor_id == rule_contractor_id:
            scoped_matches.append(item)
        elif not rule_contractor_id:
            generic_matches.append(item)

    if scoped_matches:
        _, r, hit_kws = max(scoped_matches, key=lambda x: x[0])
        return Candidate(
            article=str(r.get("article_name", "")),
            score=1.0,
            source="stage1_own_rules",
            is_own=True,
            payload={
                "keywords_raw": r.get("keywords_raw", ""),
                "matched_keywords": hit_kws,
                "match_scope": "text+contractor_id",
                "rule_contractor_id": normalize_id(r.get("contractor_id", "")),
            },
        )

    if generic_matches:
        _, r, hit_kws = max(generic_matches, key=lambda x: x[0])
        return Candidate(
            article=str(r.get("article_name", "")),
            score=0.98,
            source="stage1_own_rules",
            is_own=True,
            payload={
                "keywords_raw": r.get("keywords_raw", ""),
                "matched_keywords": hit_kws,
                "match_scope": "text_only",
                "rule_contractor_id": "",
            },
        )
    return None


def stage1_own_history(
    purpose_norm: str,
    contractor_id: str,
    history_rows: List[Dict[str, Any]],
) -> Optional[Candidate]:
    if not purpose_norm:
        return None

    hits_scoped: List[str] = []
    hits_generic: List[str] = []

    for r in history_rows:
        article = str(r.get("article_name", ""))
        if not article:
            continue
        if str(r.get("purpose_norm", "")) != purpose_norm:
            continue

        hist_contractor_id = normalize_id(r.get("contractor_id", ""))
        if contractor_id:
            if hist_contractor_id and hist_contractor_id == contractor_id:
                hits_scoped.append(article)
            elif not hist_contractor_id:
                hits_generic.append(article)
        else:
            if not hist_contractor_id:
                hits_generic.append(article)

    if hits_scoped:
        article = Counter(hits_scoped).most_common(1)[0][0]
        return Candidate(
            article=article,
            score=0.97,
            source="stage1_own_history",
            is_own=True,
            payload={"hits": len(hits_scoped), "match_scope": "purpose+contractor_id"},
        )

    if not hits_generic:
        return None

    article = Counter(hits_generic).most_common(1)[0][0]
    return Candidate(
        article=article,
        score=0.95,
        source="stage1_own_history",
        is_own=True,
        payload={"hits": len(hits_generic), "match_scope": "purpose_only"},
    )


def stage1_foreign_rules(purpose_raw: str, foreign_rule_rows: List[Dict[str, Any]]) -> List[Candidate]:
    out: List[Candidate] = []
    for r in foreign_rule_rows:
        if keyword_match(purpose_raw, str(r.get("keywords_raw", "")), str(r.get("keywords_logic", "or"))):
            article = str(r.get("article_name", ""))
            if article:
                out.append(
                    Candidate(
                        article=article,
                        score=0.75,
                        source="stage1_foreign_rules",
                        is_own=False,
                        payload={"source_client_id": r.get("client_id", "")},
                    )
                )
    return out


def top_k_by_vector(
    vec: List[float],
    rows: Iterable[Dict[str, Any]],
    vector_col: str,
    top_k: int,
    source: str,
    own_client_id: str,
) -> List[Candidate]:
    scored: List[Candidate] = []
    for r in rows:
        article = str(r.get("article_name", ""))
        if not article:
            continue
        rv = parse_vec(r.get(vector_col, ""))
        if not rv:
            continue
        score = cosine_similarity(vec, rv)
        scored.append(
            Candidate(
                article=article,
                score=score,
                source=source,
                is_own=(normalize_id(r.get("client_id", "")) == own_client_id),
                payload={"source_client_id": normalize_id(r.get("client_id", ""))},
            )
        )
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:top_k]


def map_foreign_to_own_article(
    foreign_article: str,
    own_article_vectors: Dict[str, List[float]],
    nlp,
) -> Optional[Tuple[str, float]]:
    if not own_article_vectors:
        return None

    foreign_vec = to_vec(nlp(foreign_article))
    best_article: Optional[str] = None
    best_score = -1.0
    for own_article, own_vec in own_article_vectors.items():
        if not own_vec:
            continue
        s = cosine_similarity(foreign_vec, own_vec)
        if s > best_score:
            best_score = s
            best_article = own_article

    if best_article is None:
        return None
    return (best_article, best_score)


def decide_final(
    client_id: str,
    candidates: List[Candidate],
    own_article_vectors: Dict[str, List[float]],
    nlp,
    own_similarity_threshold: float,
) -> Tuple[str, str, float, str]:
    if not candidates:
        return ("", "no_candidates", 0.0, "manual_review")

    own = [c for c in candidates if c.is_own]
    if own:
        own.sort(key=lambda c: c.score, reverse=True)
        w = own[0]
        return (w.article, w.source + "_own_priority", w.score, "existing_article")

    top = max(candidates, key=lambda c: c.score)
    mapped = map_foreign_to_own_article(top.article, own_article_vectors, nlp)
    if mapped and mapped[1] >= own_similarity_threshold:
        return (mapped[0], "stage3_foreign_to_own_mapping", mapped[1], "existing_article")

    maj = Counter([c.article for c in candidates if c.article]).most_common(1)
    if maj:
        return (maj[0][0], "stage3_majority_vote", float(maj[0][1]), "new_article_candidate")

    return ("", "stage3_failed", 0.0, "manual_review")


def classify_one(
    row: Dict[str, Any],
    rules_own: List[Dict[str, Any]],
    rules_foreign: List[Dict[str, Any]],
    hist_own: List[Dict[str, Any]],
    hist_all: List[Dict[str, Any]],
    nlp,
    top_k: int,
    vector_threshold: float,
    own_similarity_threshold: float,
    own_article_vectors: Dict[str, List[float]],
) -> Dict[str, Any]:
    purpose_raw = str(row.get("purpose_raw", ""))
    contractor_id = normalize_id(row.get("contractor_id", ""))

    purpose_clean = normalize_text(purpose_raw)
    purpose_norm = lemmatize_text(purpose_clean, nlp) if purpose_clean else ""

    candidates: List[Candidate] = []

    c1 = stage1_own_rules(purpose_raw, contractor_id, rules_own)
    if c1:
        candidates.append(c1)

    if not c1:
        c2 = stage1_own_history(purpose_norm, contractor_id, hist_own)
        if c2:
            candidates.append(c2)

    candidates.extend(stage1_foreign_rules(purpose_raw, rules_foreign))

    run_stage2 = (not candidates) or (max(c.score for c in candidates) < vector_threshold)
    if run_stage2:
        purpose_vec = to_vec(nlp(purpose_norm or purpose_clean or purpose_raw))
        h = top_k_by_vector(purpose_vec, hist_all, "purpose_vector", top_k, "stage2_history", str(row.get("client_id", "")))
        k = top_k_by_vector(purpose_vec, rules_own + rules_foreign, "keyword_vector", top_k, "stage2_keywords", str(row.get("client_id", "")))
        candidates.extend((h + k)[:top_k])

    pred_article, decision_stage, confidence, suggestion_type = decide_final(
        client_id=str(row.get("client_id", "")),
        candidates=candidates,
        own_article_vectors=own_article_vectors,
        nlp=nlp,
        own_similarity_threshold=own_similarity_threshold,
    )

    row_out = dict(row)
    row_out["purpose_norm"] = purpose_norm
    row_out["predicted_article"] = pred_article
    row_out["decision_stage"] = decision_stage
    row_out["confidence"] = confidence
    row_out["suggestion_type"] = suggestion_type
    row_out["candidates_json"] = json.dumps(
        [
            {
                "article": c.article,
                "score": c.score,
                "source": c.source,
                "is_own": c.is_own,
                "payload": c.payload,
            }
            for c in sorted(candidates, key=lambda x: x.score, reverse=True)
        ],
        ensure_ascii=False,
    )
    return row_out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Классификация входящих платежей по служебным таблицам")

    p.add_argument("--service-keywords", required=True, help="Таблица ключевиков (csv/parquet)")
    p.add_argument("--service-history", required=True, help="Таблица истории (csv/parquet)")
    p.add_argument("--incoming-csv", required=True, help="CSV с новыми платежами")
    p.add_argument("--out", required=True, help="Файл результатов (csv/parquet)")

    p.add_argument("--incoming-sep", default=",", help="Разделитель входного CSV")
    p.add_argument("--client-col", default="accounts__user_id")
    p.add_argument("--purpose-col", default="purpose")
    p.add_argument("--counterparty-col", default="counterparties__name")
    p.add_argument("--direction-col", default="expenditure")
    p.add_argument("--contractor-id-col", default="contractor_id")
    p.add_argument("--payment-id-col", default="id")
    p.add_argument("--payment-date-col", default="date")

    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--vector-threshold", type=float, default=0.82)
    p.add_argument("--own-similarity-threshold", type=float, default=0.80)

    return p


def main() -> None:
    args = build_parser().parse_args()

    ensure_spacy_model()
    import spacy  # type: ignore

    nlp = spacy.load("ru_core_news_sm")

    kw_df = load_df(Path(args.service_keywords))
    hist_df = load_df(Path(args.service_history))
    incoming_df = load_df(Path(args.incoming_csv), csv_sep=args.incoming_sep)

    if "client_id" in kw_df.columns:
        kw_df["client_id"] = kw_df["client_id"].map(normalize_id)
    if "contractor_id" in kw_df.columns:
        kw_df["contractor_id"] = kw_df["contractor_id"].map(normalize_id)
    kw_df["direction"] = kw_df["direction"].map(normalize_direction) if "direction" in kw_df.columns else ""

    if "client_id" in hist_df.columns:
        hist_df["client_id"] = hist_df["client_id"].map(normalize_id)
    if "contractor_id" in hist_df.columns:
        hist_df["contractor_id"] = hist_df["contractor_id"].map(normalize_id)
    hist_df["direction"] = hist_df["direction"].map(normalize_direction) if "direction" in hist_df.columns else ""

    incoming = incoming_df.copy()
    incoming["client_id"] = incoming[args.client_col].map(normalize_id)
    incoming["payment_id"] = incoming[args.payment_id_col].fillna("").astype(str) if args.payment_id_col in incoming.columns else ""
    incoming["payment_date"] = incoming[args.payment_date_col].fillna("").astype(str) if args.payment_date_col in incoming.columns else ""
    incoming["purpose_raw"] = incoming[args.purpose_col].fillna("").astype(str)
    incoming["counterparty"] = incoming[args.counterparty_col].fillna("").astype(str) if args.counterparty_col in incoming.columns else ""
    incoming["direction"] = incoming[args.direction_col].map(normalize_direction) if args.direction_col in incoming.columns else ""
    incoming["contractor_id"] = incoming[args.contractor_id_col].map(normalize_id) if args.contractor_id_col in incoming.columns else ""

    kw_rows = kw_df.to_dict(orient="records")
    hist_rows = hist_df.to_dict(orient="records")

    by_client_rules: Dict[str, List[Dict[str, Any]]] = {}
    for r in kw_rows:
        by_client_rules.setdefault(normalize_id(r.get("client_id", "")), []).append(r)

    by_client_hist: Dict[str, List[Dict[str, Any]]] = {}
    for r in hist_rows:
        by_client_hist.setdefault(normalize_id(r.get("client_id", "")), []).append(r)

    kw_rows_by_direction: Dict[str, List[Dict[str, Any]]] = {}
    for r in kw_rows:
        kw_rows_by_direction.setdefault(normalize_direction(r.get("direction", "")), []).append(r)

    hist_rows_by_direction: Dict[str, List[Dict[str, Any]]] = {}
    for r in hist_rows:
        hist_rows_by_direction.setdefault(normalize_direction(r.get("direction", "")), []).append(r)

    rows = incoming.to_dict(orient="records")
    try:
        from tqdm import tqdm  # type: ignore

        iterator = tqdm(rows, total=len(rows), desc="classifying incoming")
    except Exception:
        iterator = rows

    out_rows: List[Dict[str, Any]] = []
    for row in iterator:
        cid = normalize_id(row.get("client_id", ""))
        direction = normalize_direction(row.get("direction", ""))

        rules_own_all = by_client_rules.get(cid, [])
        hist_own_all = by_client_hist.get(cid, [])

        if direction:
            rules_own = [r for r in rules_own_all if normalize_direction(r.get("direction", "")) in {"", direction}]
            hist_own = [r for r in hist_own_all if normalize_direction(r.get("direction", "")) in {"", direction}]

            kw_pool = kw_rows_by_direction.get(direction, []) + kw_rows_by_direction.get("", [])
            hist_pool = hist_rows_by_direction.get(direction, []) + hist_rows_by_direction.get("", [])

            rules_foreign = [r for r in kw_pool if normalize_id(r.get("client_id", "")) != cid]
            hist_all = hist_pool
        else:
            rules_own = rules_own_all
            hist_own = hist_own_all
            rules_foreign = [r for r in kw_rows if normalize_id(r.get("client_id", "")) != cid]
            hist_all = hist_rows

        own_article_vectors: Dict[str, List[float]] = {}
        for r in rules_own + hist_own:
            a = str(r.get("article_name", ""))
            if not a:
                continue
            v = parse_vec(r.get("article_vector", ""))
            if v and a not in own_article_vectors:
                own_article_vectors[a] = v

        out_rows.append(
            classify_one(
                row=row,
                rules_own=rules_own,
                rules_foreign=rules_foreign,
                hist_own=hist_own,
                hist_all=hist_all,
                nlp=nlp,
                top_k=args.top_k,
                vector_threshold=args.vector_threshold,
                own_similarity_threshold=args.own_similarity_threshold,
                own_article_vectors=own_article_vectors,
            )
        )

    pd = ensure_pandas()
    out_df = pd.DataFrame(out_rows)
    save_df(out_df, Path(args.out))

    print(
        json.dumps(
            {
                "status": "ok",
                "rows_in": int(len(incoming)),
                "rows_out": int(len(out_df)),
                "output": args.out,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
