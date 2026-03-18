# %%
#!/usr/bin/env python3

# прогнозирование платежей c помощью CatBoost (месячная агрегация)

# %%
import sys
import subprocess

if __name__ == "__main__":
    # установка и импорт нужной версии PyTorch
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch==2.6.0",
            "--index-url", "https://download.pytorch.org/whl/cu118"],
            stdout=subprocess.DEVNULL)
    except:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch==2.6.0"],
            stdout=subprocess.DEVNULL)
    try:
        import numpy as np
        if np.__version__ != "1.26.4":
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.26.4", "--force-reinstall"])
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.26.4", "--force-reinstall"])
        import numpy as np

    # установка и импорт нужной версии spacy и языковой модели
    try:
        import spacy
        from spacy.util import is_package
        if spacy.__version__ != "3.6.1":
            subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy==3.6.1", "--force-reinstall"], stdout=subprocess.DEVNULL)
            import importlib
            importlib.reload(spacy)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy==3.6.1", "--force-reinstall"], stdout=subprocess.DEVNULL)
        import spacy
        from spacy.util import is_package

    MODEL_WHL = (
        "https://github.com/explosion/spacy-models/releases/download/"
        "ru_core_news_sm-3.6.0/ru_core_news_sm-3.6.0-py3-none-any.whl"
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

# %%
# 📦 стандартные библиотеки Python
import os
import random
import re
import multiprocessing
import logging
import requests

# 🧮 научные и табличные библиотеки
import numpy as np
import pandas as pd
import scipy

# ⏳ прогресс-бары
import tqdm as tqdm_lib
from tqdm import tqdm

# 🧠 обработка текста и NLP
import spacy
try:
    import transformers
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"], stdout=subprocess.DEVNULL)
    import transformers
    from transformers import AutoModel, AutoTokenizer

# 🤖 pyTorch
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

# 🛠 scikit-learn: пайплайны, препроцессинг, модели
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# 🧠 CatBoost
try:
    import catboost
    from catboost import CatBoostRegressor, utils
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost==1.2.7"], stdout=subprocess.DEVNULL)
    import catboost
    from catboost import CatBoostRegressor, utils

# обработка warning
import warnings

# 📉 PCA
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
# %%

# блок с функциями
def seed_all(seed_value):
    logger.debug(f"🔍 Установка глобального seed={seed_value} для воспроизводимости.")
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.benchmark = False

# создаем функцию очистки текста
def _clean_single_text(text):
    return re.sub(r"[^\w\s]", " ", text.lower())

# создаем функцию предобработки текста
def preprocess_texts_optimized(texts, nlp_model_name,
                               batch_size_cpu=256,
                               num_processes_for_cleaning=-1,
                               num_processes_for_spacy_cpu=-1):
    
    logger.debug(f"🔍 Запуск предобработки для {len(texts)} текстов...")
    
    # предварительная очистка текстов
    cleaned_texts = [_clean_single_text(text) for text in tqdm(texts, desc="Очистка")]

    # spaCy и лемматизация
    nlp = None
    processed_lemmas = []
    
    # загрузка NLP модели
    logger.debug(f"🔍 Загрузка spaCy модели: '{nlp_model_name}'.")
    nlp = spacy.load(nlp_model_name)

    # используем n_process для параллелизации
    if num_processes_for_spacy_cpu == -1:
        cpu_count = os.cpu_count()
        num_processes_for_spacy_cpu = max(1, cpu_count - 1)
    
    logger.debug(f"🔍 Лемматизация будет выполнена в {num_processes_for_spacy_cpu} потоках.")
    
    for doc in tqdm(nlp.pipe(cleaned_texts, batch_size=batch_size_cpu, n_process=num_processes_for_spacy_cpu), total=len(cleaned_texts), desc="Лемматизация (CPU)"):
        lemmas = [token.lemma_ for token in doc]
        processed_lemmas.append(" ".join(lemmas))
    
    logger.debug(f"🔍 Предобработка завершена. Обработано {len(processed_lemmas)} текстов.")
    return processed_lemmas

# создаем функцию для получения усредненного эмбеддинга текста
def get_embeddings_batch(texts, tokenizer, model, device, batch_size=64):
    texts = list(texts)
    embeddings = []

    logger.debug(f"🔍 Начало генерации эмбеддингов для {len(texts)} текстов на устройстве '{device}'.")
    for i in tqdm(range(0, len(texts), batch_size), desc="Генерация эмбеддингов"):
        batch_texts = texts[i:i+batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # переносим каждый тензор на устройство
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # берем attention mask (1 — реальные токены, 0 — паддинг)
        mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size())
        masked_embeddings = outputs.last_hidden_state * mask

        # считаем среднее только по непаддинговым токенам
        summed = masked_embeddings.sum(dim=1)
        counts = mask.sum(dim=1)
        mean_pooled = summed / counts

        embeddings.extend(mean_pooled.cpu().numpy())
    
    logger.debug(f"🔍 Генерация эмбеддингов завершена")
    
    return embeddings

# создаем функцию скользящего среднего (адаптирована под месяцы) и лагов
def rolling_mean(group, rolling_window, lags):
    group = group.sort_values("date")
    
    # скользящее среднее
    group["rolling_month_payments"] = (
        group["month_payments_sum"]
        .shift(1)
        .rolling(window=rolling_window, min_periods=1)
        .mean()
    )
    
    # лаги (по месяцам: 1, 2, 3, 6, 12)
    for lag in lags:
        group[f"month_payments_sum_lag_{lag}"] = group["month_payments_sum"].shift(lag)

    return group

# создаем функцию заполнения пропусков нулями в заданном временном ряду (адаптирована под месяцы)
def fill_missing_dates(group):
    current_user_id = group.name

    end_date = pd.Timestamp.today().replace(day=1) - pd.DateOffset(months=1)

    full_range = pd.date_range(group['date'].min(), end_date, freq='MS')
        
    original_rows = len(group)
    new_rows = len(full_range)

    logger.debug(
        f"🔍 Заполнение месяцев для user_id='{current_user_id}'. "
        f" Диапазон: {full_range.min().date()} - {full_range.max().date()}. "
        f" Строк до: {original_rows}, строк после: {new_rows}."
    )

    group = group.set_index("date").reindex(full_range)
    group["user_id"] = current_user_id

    # пропуски заполним нулями
    for col in group.columns:
        if col not in ["user_id"]:
            group[col] = group[col].fillna(0)

    return group.reset_index().rename(columns={"index": "date"})

# создаем функцию MASE
def mase(y_true, y_pred, y_train, seasonality=12): # Seasonality 12 для месячных
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_train = np.asarray(y_train)
    
    # проверка на достаточность данных для сезонности
    if len(y_train) <= seasonality:
        scale = np.mean(np.abs(np.diff(y_train))) # Если данных мало, берем лаг 1
    else:
        scale = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
        
    if scale == 0:
        # если сезонная разница равна 0
        scale = np.mean(np.abs(np.diff(y_train))) + 1e-8
        
    if scale == 0:
         logger.error("❌ Деление на ноль в MASE даже после fallback!")
         return np.inf

    return np.mean(np.abs(y_true - y_pred)) / scale

# создаем функцию проверки начальных значений на нули или слишком разреженные данные
def start_date_define(df, target_col, window_size=3, min_nonzero=1, num_windows=1): # было 6-3- на днях
    # Окна по 6 месяцев, нужно чтобы хотя бы 3 были не нулевыми
    series = df[target_col]
    nz = (series > 0).astype(int).values
    n = len(nz)

    # для каждого окна проверяем, хватает ли ненулевых значений
    cond = [nz[i:i+window_size].sum() >= min_nonzero for i in range(n - window_size + 1)]

    count = 0
    for i, val in enumerate(cond):
        if val:
            count += 1
            if count >= num_windows:
                start_idx = i - num_windows + 1
                logger.info(f"ℹ️ Подходящая начальная дата по фильтру: {df.index[start_idx]}. Данные до этой даты будут отброшены.")
                logger.debug(f"🔍 Первая дата с данными во всем ряде: {df.index.min()}")
                return df.iloc[start_idx:]
        else:
            count = 0

    logger.warning(f"⚠️ Не найдено подходящей начальной даты, удовлетворяющей критериям. Временной ряд будет пропущен.")
    return None  

# %%
# устанавливаем значение для случайных значений и другие константы
RANDOM_STATE = 42
ROLLING_WINDOW = 3 
LAGS = [1, 2, 3, 6, 12] 
DATA_DIR = os.getenv("DATA_DIR", ".")

# %%
# настраиваем логгер
LOG_PATH = os.path.join(DATA_DIR, "monthly_forecast_prod.log")
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)

# файл
file_handler = logging.FileHandler(LOG_PATH, mode="a")
file_handler.setLevel(logging.INFO)
file_format = logging.Formatter(
    "%(asctime)s - %(levelname)-8s - %(funcName)-20s: %(lineno)-4d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_format)

# консоль
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_format = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
console_handler.setFormatter(console_format)

# ⚠️ защита от дублирования хендлеров
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# %%
if __name__ == "__main__":
    logger.info(f"✅ Запуск скрипта")

    # %%

    # вынесем блок с настройками и проверками версий
    versions = [
    f"{'':21}numpy           {np.__version__}",
    f"{'':21}pandas          {pd.__version__}",
    f"{'':21}tqdm            {tqdm_lib.__version__}",
    f"{'':21}torch           {torch.__version__}",
    f"{'':21}sklearn         {sklearn.__version__}",
    f"{'':21}scipy           {scipy.__version__}",
    f"{'':21}spacy           {spacy.__version__}",
    f"{'':21}transformers    {transformers.__version__ if transformers else 'not installed'}",
    f"{'':21}catboost        {catboost.__version__ if catboost else 'not installed'}",
    f"{'':21}CUDA available  {torch.cuda.is_available()}"
    ]

    logger.info(f"🔍 Версии библиотек:\n" + "\n".join(versions))

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:.2f}".format)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # %%
    # импортируем данные
    # настраиваем доступы по API через профиль

    profile = os.getenv("API_PROFILE", "prod").lower()
    
    if profile == "prod":
        API_URL_DOWN = os.getenv(
            "PAYMENTS_API_URL_DOWN",
            "https://api.lemonpie.tech/api/payments/ai"
        )
        API_TOKEN_DOWN = os.getenv("PAYMENTS_API_TOKEN_DOWN")

        API_URL_UP = os.getenv(
            "PAYMENTS_API_URL_UP",
            "https://api.lemonpie.tech/api/forecasts"
        )
        API_TOKEN_UP = os.getenv("PAYMENTS_API_TOKEN_UP")

    elif profile == "dev":
        API_URL_DOWN = os.getenv("API2_URL_DOWN")
        API_TOKEN_DOWN = os.getenv("API2_TOKEN_DOWN")

        API_URL_UP = os.getenv("API2_URL_UP")
        API_TOKEN_UP = os.getenv("API2_TOKEN_UP")

    else:
        raise RuntimeError(f"Unknown API_PROFILE={profile}")

    if not API_URL_DOWN:
        raise RuntimeError(f"{profile}: API_URL_DOWN is not set")
    if not API_TOKEN_DOWN:
        raise RuntimeError(f"{profile}: API_TOKEN_DOWN is not set")

    if not API_URL_UP:
        raise RuntimeError(f"{profile}: API_URL_UP is not set")
    if not API_TOKEN_UP:
        raise RuntimeError(f"{profile}: API_TOKEN_UP is not set")

    headers_down = {
        "Authorization": f"Bearer {API_TOKEN_DOWN}"
    }
    headers_up = {
        "Authorization": f"Bearer {API_TOKEN_UP}",
        "Content-Type": "application/json",
    }

    logger.info(f"API-profile [{profile}] GET-запрос: {API_URL_DOWN}")
    logger.info(f"API-profile [{profile}] POST-запрос: {API_URL_UP}")
    ##

    path1 = "3try_data_with_purpose_mean.parquet"

    if os.path.exists(path1):
        data = pd.read_parquet(path1)
        logger.info(f"ℹ️ Данные загружены из файла: {path1}")
    else:
        logger.info('ℹ️ Файл с данными не обнаружен. Начинаем загрузку данных с сервера...')

        params = {
            "limit": 5000,
            "page": 1,
            "expenditure": "incoming"
        }

        all_data = []

        with tqdm(desc="Загружено страниц", unit=" стр", dynamic_ncols=True) as pbar:
            while True:
                response = requests.get(API_URL_DOWN, headers=headers_down, params=params,timeout=30)
                if response.status_code != 200:
                    logger.error(f"❌ Ошибка загрузки данных с сервера: {response.status_code}")
                    break

                result = response.json()
                page_data = result.get("data", [])
                if not page_data:
                    break

                all_data.extend(page_data)
                if len(page_data) < params["limit"]:
                    break

                params["page"] += 1
                pbar.update(1)

        data_full = pd.json_normalize(all_data, sep="__")
        data_full.to_parquet(os.path.join(DATA_DIR, "data_full.parquet"), index=False)
        logger.info(f"ℹ️ Данные загружены с сервера. Количество записей: {len(data_full)}")

        data_full["date"] = pd.to_datetime(data_full["date"], yearfirst=True, errors='coerce')
        print("Proverka data_full_1:", data_full["date"].dt.day.value_counts().sort_index())

        # заполняем accounts__user_id значениями из других столбцов
        data_full["accounts__user_id"] = (
            data_full["accounts__user_id"]
            .fillna(data_full["articles__user_id"])
            .fillna(data_full["projects__user_id"])
            .fillna(data_full["counterparties__user_id"])
            .fillna(data_full["robots__user_id"])
            .fillna(data_full["article_alternative_names__user_id"])
        )
        print("Proverka data_full_2:", data_full["date"].dt.day.value_counts().sort_index())

        # отфильтруем данные
        mask_interest = data_full["purpose"].str.contains("процент", case=False, na=False)
        mask_free = data_full["purpose"].str.contains("беспроцент", case=False, na=False)

        data_actual_id_wodepo = data_full[
            ~(mask_interest & ~mask_free)
            & ~data_full["purpose"].str.contains("вклад", case=False, na=False)
            & ~data_full["purpose"].str.contains("депози", case=False, na=False)
            & ~data_full["purpose"].str.contains("собствен", case=False, na=False)
        ]
        
        # удаляем столбцы
        data = data_actual_id_wodepo.drop(
            [
                "robot_id",
                "accounts__id",
                "articles__id",
                "articles__user_id",
                "projects__id",
                "projects__user_id",
                "counterparties__id",
                "counterparties__user_id",
                "robots__user_id",
                "article_alternative_names__user_id",
                "articles__name",
                "projects__name",
                "counterparties__name",
                "uc__uc_id"
            ],
            axis=1,
        )

        data[
            [
                "articles__parent_id",
                "projects__parent_id",
                "counterparties__parent_id",
                "robots__id",
            ]
        ] = (
            data[
                [
                    "articles__parent_id",
                    "projects__parent_id",
                    "counterparties__parent_id",
                    "robots__id",
                ]
            ]
            .fillna(0)
            .astype("int64")
        )

        data["purpose"] = data["purpose"].fillna("missing")

        data = data.rename(columns={"accounts__user_id": "user_id"})
        data["user_id"] = data["user_id"].fillna(0).astype("int64")

        # закодируем текстовое поле
        logger.info("ℹ️ Закодируем текстовое поле")

        data["clean_purpose"] = preprocess_texts_optimized(texts=data["purpose"],nlp_model_name="ru_core_news_sm")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")
        model = model.to(device)

        data["purpose_emb"] = get_embeddings_batch(data["clean_purpose"], tokenizer, model, device)

        data["purpose_mean"] = data["purpose_emb"].apply(lambda x: float(np.mean(x)))

        batch_size = 10_000
        scaler = StandardScaler()
        ipca = IncrementalPCA(n_components=3)

        for i in tqdm(range(0, len(data), batch_size), desc="Обучение StandardScaler"):
            batch = np.vstack(data["purpose_emb"].iloc[i:i+batch_size])
            scaler.partial_fit(batch)

        for i in tqdm(range(0, len(data), batch_size), desc="Обучение IncrementalPCA"):
            batch = np.vstack(data["purpose_emb"].iloc[i:i+batch_size])
            batch_scaled = scaler.transform(batch)
            ipca.partial_fit(batch_scaled)

        transformed_batches = []
        for i in tqdm(range(0, len(data), batch_size), desc="Масштабируем эмбеддинги"):
            batch = np.vstack(data["purpose_emb"].iloc[i:i+batch_size]).astype(np.float32)
            batch_scaled = scaler.transform(batch)
            transformed_batches.append(ipca.transform(batch_scaled))

        purpose_pca_features = np.vstack(transformed_batches)

        pca_column_names = [f"purpose_pca_{i+1}" for i in range(3)]
        data[pca_column_names] = purpose_pca_features

        data.drop(columns=["purpose", "clean_purpose", "purpose_emb"], inplace=True)

        data.to_parquet(path1, index=False)
        logger.info(f'ℹ️ Тексты закодированы')

    # %%
    # конвертируем дату в datetime
    data["date"] = pd.to_datetime(data["date"], yearfirst=True, errors='coerce')

    # и убираем записи из будущего и далекого прошлого, меньше нуля (и такое бывает) и без привязки к фондам (user_id == 0)
    yesterday = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    data = data[data["date"] <= yesterday]
    data = data[data["payments_amount"] > 0]
    data = data[data["date"] >= "2015-01-01"]
    data = data[data["user_id"] != 0]

    data = data.drop("id", axis=1)

    # %%
    # закодируем данные с помощью frequency_encoding по каждому user_id
    logger.info('ℹ️ Кодируем категориальные данные')

    id_cols = [
        "account_id",
        "contractor_id",
        "article_id",
        "project_id",
        "counterpartie_id",
        "donor_id",
        "donor_cat_id",
        "articles__parent_id",
        "projects__parent_id",
        "counterparties__parent_id",
        "robots__id",
    ]

    data_fe = data.copy()
    
    data_fe["user_total_counts_col"] = data_fe.groupby("user_id")["user_id"].transform("size")
    
    for col in id_cols:
        encoded_col_name = f"{col}_ufreq"
        numerator = data_fe.groupby(["user_id", col])[col].transform("size")
        data_fe[encoded_col_name] = numerator.astype(float) / data_fe["user_total_counts_col"].astype(float)

    data_fe.drop(columns=["user_total_counts_col"], inplace=True)

    # %%
    # агрегация платежей по МЕСЯЦАМ и пользователям
    logger.info('ℹ️ Агрегируем атомарные данные до уровня месяцев')
    
    data_fe["month_date"] = data_fe["date"].dt.to_period("M").dt.to_timestamp()

    nunique_cols = [
        "account_id",
        "contractor_id",
        "article_id",
        "project_id",
        "counterpartie_id",
        "donor_id",
        "donor_cat_id",
        "articles__parent_id",
        "projects__parent_id",
        "counterparties__parent_id",
        "robots__id",
    ]

    purpose_mean_cols = ["purpose_mean", "purpose_pca_1", "purpose_pca_2", "purpose_pca_3"]
    col_ufreq = [col for col in data_fe.columns if col.endswith("_ufreq")]

    data_monthly_agg = (
        data_fe.groupby(["user_id", "month_date"])
        .agg(
            **{f"month_{col}_nunique": (col, "nunique") for col in nunique_cols},
            month_payments_sum=("payments_amount", "sum"),
            month_payments_count=("payments_amount", "count"),
            month_min_payment=("payments_amount", "min"),
            month_max_payment=("payments_amount", "max"),
            **{f"month_{col}": (col, "mean") for col in purpose_mean_cols},
            **{f"{col}_mean": (col, "mean") for col in col_ufreq},
            **{f"{col}_std": (col, "std") for col in col_ufreq},
        )
        .reset_index()
        .rename(columns={"month_date": "date"})
    )

    ## сохраним фактические данные
    actual_monthly_payments = (
        data_monthly_agg[["user_id", "date", "month_payments_sum"]]
        .sort_values(["user_id", "date"])
        .groupby("user_id", group_keys=False)
        .apply(
            lambda df: (
                df
                .set_index("date")
                .asfreq("MS")
                .assign(user_id=df["user_id"].iloc[0])
                .fillna({"month_payments_sum": 0.0})
            )
        )
        .reset_index()
    )
    actual_monthly_payments.to_csv(os.path.join(DATA_DIR, "actual_monthly_payments.csv"),index=False)
    ##

    std_cols = [col for col in data_monthly_agg.columns if col.endswith("_std")]
    data_monthly_agg[std_cols] = data_monthly_agg[std_cols].fillna(0)

    for col in data_monthly_agg.columns:
        if col.endswith("_ufreq_mean") or col.endswith("_ufreq_std"):
            data_monthly_agg[col] = data_monthly_agg[col].round(6)
        elif col in ["month_payments_sum", "month_min_payment", "month_max_payment"]:
            data_monthly_agg[col] = data_monthly_agg[col].round(2)

    # %%
    # заполним пропущенные МЕСЯЦЫ
    logger.info('ℹ️ Заполняем пропуски в агрегированных данных (месячный временной ряд) нулями')
   
    data_monthly_agg_filled = data_monthly_agg.groupby("user_id", group_keys=False).apply(fill_missing_dates)

    logger.info('ℹ️ Добавляем временные признаки (месяц, квартал)')
    data_monthly_agg_filled["month_of_year"] = data_monthly_agg_filled["date"].dt.month
    data_monthly_agg_filled["month_sin"] = np.sin(2 * np.pi * data_monthly_agg_filled["month_of_year"] / 12).round(6)
    data_monthly_agg_filled["month_cos"] = np.cos(2 * np.pi * data_monthly_agg_filled["month_of_year"] / 12).round(6)
    data_monthly_agg_filled["quarter"] = data_monthly_agg_filled["date"].dt.quarter

    # %%
    # добавим скользящее среднее и лаги
    logger.info('ℹ️ Добавляем скользящие средние и лаги по целевому признаку')

    data_monthly_agg_filled_added = data_monthly_agg_filled.groupby(
        "user_id", group_keys=False
    ).apply(lambda group: rolling_mean(group, rolling_window=ROLLING_WINDOW, lags=LAGS))

    data_monthly_agg_filled_added = data_monthly_agg_filled_added.fillna(0)

    # %%
    # преобразуем дату в индекс
    data_final = data_monthly_agg_filled_added.set_index("date", drop=True)
    data_final.index.name = None

    # %%
    # удаляем лишние колонки
    cols_to_drop_6 = [
        "month_robots__id_nunique",
        "month_counterpartie_id_nunique",
        "month_projects__parent_id_nunique",
        "month_donor_id_nunique",
        "month_articles__parent_id_nunique",
        "month_counterparties__parent_id_nunique",
        "month_account_id_nunique",
        "month_article_id_nunique",
        "month_project_id_nunique",
        "article_id_ufreq_std",
        "projects__parent_id_ufreq_std",
        "counterpartie_id_ufreq_std",
        "donor_id_ufreq_std",
        "account_id_ufreq_std",
        "counterparties__parent_id_ufreq_std",
        "project_id_ufreq_std",
        "articles__parent_id_ufreq_std",
        "robots__id_ufreq_std",
        "donor_id_ufreq_mean",
        "counterpartie_id_ufreq_mean",
        "projects__parent_id_ufreq_mean",
        "account_id_ufreq_mean",
        "robots__id_ufreq_mean",
        "articles__parent_id_ufreq_mean",
        "counterparties__parent_id_ufreq_mean",
    ]

    data_final = data_final.drop(columns=cols_to_drop_6)
    data_final.to_csv(os.path.join(DATA_DIR, "data_final.csv"))
    
    logger.info('ℹ️ Данные обработаны и подготовлены к обучению')

    # %%
    # запускаем обучение моделей
    data_final = data_final.sort_index()

    # подсчёт количества активных месяцев
    months_per_user_id = data_final[data_final["month_payments_sum"] != 0].groupby("user_id")['month_payments_sum'].count()

    # проверка последних 3 месяцев на наличие данных
    months_per_user_nonzero = (
        data_final
        .sort_index()
        .groupby('user_id')
        .apply(lambda x: (x.iloc[-4:]['month_payments_sum'] != 0).sum())
    )

    # фильрация
    funds = months_per_user_id[
        (months_per_user_id >= 6) &
        (months_per_user_nonzero > 0)
    ].index.tolist()

    
    logger.info(f'ℹ️ Фильтрация по количеству активных месяцев и наличию платежей проведена, отобраны следующие фонды: {funds}')

    # %%
    # запускаем общий цикл обучения
    logger.info('ℹ️ Запущен общий цикл обучения и прогнозирования по списку отобранных фондов')

    mode = "production"  # production/train_val
    logger.info(f"ℹ️ Режим прогнозирования: {mode}")

    if mode == "production":
        forecast_catboost_rows = []

    for fund in tqdm(funds):

        # %%
        # выберем фонд
        FUND_ID = fund
        logger.info(f"🔹 FUND_ID: {FUND_ID}")

        # устанавливаем сиды
        user_specific_seed = RANDOM_STATE + int(FUND_ID) 
        seed_all(user_specific_seed)

        data_fund = data_final[data_final["user_id"] == FUND_ID].copy()

        if not data_fund.index.is_monotonic_increasing:
            data_fund = data_fund.sort_index()

        # проверяем ряд на стартовые нули
        data_fund = start_date_define(data_fund, target_col="month_payments_sum")

        if data_fund is None:
            logger.warning(f"⚠️ Фонд {FUND_ID} пропускаем — ряд данных слишком разреженный")
            continue

        data_fund.drop(["user_id"], axis=1, inplace=True)

        # %%
        # сглаживание выбросов
        TRESHOLD_CONST = 0.95 
        threshold = data_fund["month_payments_sum"].quantile(TRESHOLD_CONST)
        data_fund.loc[data_fund["month_payments_sum"] > threshold, "month_payments_sum"] = data_fund["month_payments_sum"].median()

        # оставляем только полные месяцы (все, что меньше начала текущего месяца)
        current_month_start = pd.Timestamp.today().replace(day=1)
        
        data_fund = data_fund[data_fund.index < current_month_start]

        # %%
        # создаём версию со сдвигом
        data_fund_shifted = data_fund.copy()
        data_fund_shifted["target_shifted"] = data_fund_shifted["month_payments_sum"].shift(-1)

        # разделение на train и test
        if mode == "train_val":
            # Для валидации удаляем будущее (где таргет NaN), так как на нем нельзя проверить качество
            data_fund_shifted = data_fund_shifted[:-1]
            
            # тест 2 месяца
            last_available_date = data_fund_shifted.index.max()
            test_start_date = last_available_date - pd.DateOffset(months=1)
            
            train_data_fund = data_fund_shifted.loc[: test_start_date - pd.DateOffset(months=1)]
            test_data_fund = data_fund_shifted.loc[test_start_date:]

        elif mode == "production": # <-- не забудьте двоеточие
            # train - всё, где известен таргет (история)
            train_data_fund = data_fund_shifted.dropna(subset=["target_shifted"])
            # test - последняя точка, для которой нужно предсказать будущее
            test_data_fund = data_fund_shifted.tail(1)

        logger.debug(f'🔍 Размеры: train={train_data_fund.shape}, test={test_data_fund.shape}')
        
        # %%
        # выделяем категоральные признаки
        cat_features = ["month_of_year"]

        # формируем X_train и y_train
        X_train = train_data_fund.drop(["month_payments_sum", "target_shifted"], axis=1)
        y_train = train_data_fund["target_shifted"]

        # формируем X_test и y_test
        if mode == "train_val":
            X_test = test_data_fund.drop(["month_payments_sum", "target_shifted"], axis=1)
            y_test = test_data_fund["target_shifted"]
        elif mode == "production":
            X_test = test_data_fund.drop(["month_payments_sum"], axis=1)

        # %%
        # Catboost
        logger.info("ℹ️ Обучение Catboost и прогнозирование начаты")

        n_splits = 3
        # проверка фолдов
        if len(X_train) > n_splits + 1:
            tss = TimeSeriesSplit(n_splits=n_splits)
        else:
            tss = None

        pipe_final = Pipeline([
            ("model", CatBoostRegressor(
                silent=True, 
                random_seed=RANDOM_STATE, 
                cat_features=cat_features,
                thread_count=1,
                allow_writing_files=False
            ))
        ])

        param_grid = [{
            "model__iterations": [100, 300, 500],
            "model__depth": [1, 3,4, 6],
            "model__learning_rate": [0.01, 0.05, 0.1,0.2],
            "model__l2_leaf_reg": [3, 10, 20, 30],
            "model__random_strength": [1, 5,10]
        }]

        # GridSearch или прямой Fit
        if tss:
            grid_search = GridSearchCV(
                pipe_final,
                param_grid=param_grid,
                cv=tss,
                scoring="neg_mean_squared_error",
                n_jobs=1
            )
            grid_search.fit(X_train, y_train)

            final_model_cb = grid_search.best_estimator_
        else:
            final_model_cb = pipe_final.fit(X_train, y_train)
        
        logger.debug(f"Лучшие гиперпараметры: {grid_search.best_params_}")

        # прогноз на Test
        y_test_pred_cb_val = np.clip(final_model_cb.predict(X_test), 0, None)
        
        # CatBoost метрики
        if mode == "train_val":
            # если предсказание скаляр, оборачиваем в список, иначе оставляем как массив
            preds = [y_test_pred_cb_val] if np.isscalar(y_test_pred_cb_val) else y_test_pred_cb_val
            
            rmse_test_cb = np.sqrt(mean_squared_error(y_test, preds))
            mase_test_cb = mase(y_test, preds, y_train, 12)

            # расчет метрики для медианы
            # считаем медиану по тренировочным данным (истории)
            if len(train_data_fund) > 0:
                median_value = train_data_fund["month_payments_sum"].median()
            else:
                median_value = 0

            # создаем "прогноз" из медианы той же длины, что и тест
            median_preds = [median_value] * len(y_test)
            
            # считаем ошибку медианы
            rmse_median = np.sqrt(mean_squared_error(y_test, median_preds))
            mase_median = mase(y_test, median_preds, y_train, 12)
            
        # для продакшена или сохранения берем первое значение (если нужно) или весь массив
        if mode == "production":
            # в продакшене X_test всегда одна строка
            if isinstance(y_test_pred_cb_val, np.ndarray):
                y_test_pred_cb_val = y_test_pred_cb_val[0]

        logger.info("ℹ️ Обучение Catboost и прогнозирование завершены")
        
        # %%
        # cохранение результатов
        if mode == "train_val":
            path3 = os.path.join(DATA_DIR, "comparison_train_val_monthly.csv")
            if os.path.exists(path3):
                comparison = pd.read_csv(path3)
            else:
                comparison = pd.DataFrame(columns=["fund_id"])

            if FUND_ID in comparison["fund_id"].values:
                idx = comparison[comparison["fund_id"] == FUND_ID].index
                comparison.loc[idx, "catboost_rmse"] = rmse_test_cb
                comparison.loc[idx, "median_rmse"] = rmse_median
                comparison.loc[idx, "catboost_mase"] = mase_test_cb
                comparison.loc[idx, "median_mase"] = mase_median
                # считаем выигрыш: если > 0, то CatBoost лучше, если < 0, то медиана лучше.
                comparison.loc[idx, "catboost_gain"] = rmse_median - rmse_test_cb 
            else:
                new_row = pd.DataFrame({
                    "fund_id": [FUND_ID],
                    "catboost_rmse": [rmse_test_cb],
                    "median_rmse": [rmse_median],
                    "catboost_mase": [mase_test_cb],
                    "median_mase": [mase_median],
                    "catboost_gain": [rmse_median - rmse_test_cb]
                })
                comparison = pd.concat([comparison, new_row], ignore_index=True)
            comparison.to_csv(path3, index=False)

        elif mode == "production":
            predict_date = data_fund.index[-1] + pd.DateOffset(months=1)
            forecast_date = (predict_date.replace(day=1).date() + pd.Timedelta(days=1)).isoformat()
            
            forecast_catboost_rows.append({
                "user_id": FUND_ID,
                "date": forecast_date,
                "amount": round(float(y_test_pred_cb_val), 2),
                "info": "catboost"
            })

    # %%
    # собираем прогнозы для production
    if mode == "production":
        
        # CatBoost прогнозы
        forecast_catboost_df = pd.DataFrame(forecast_catboost_rows)
        #forecast_catboost_df.to_csv(os.path.join(DATA_DIR, "forecasts_catboost_monthly.csv"),index=False)
        path = os.path.join(DATA_DIR, "forecasts_catboost_monthly.csv")
        forecast_catboost_df.to_csv(path, mode="a", header=not os.path.exists(path),index=False)
        forecast_final = forecast_catboost_df.copy()
        
        # медианные прогнозы для остальных фондов
        predicted_users = set(forecast_final["user_id"].unique())
        all_users = set(data_final["user_id"].unique())
        other_users = all_users - predicted_users

        median_rows = []
        for user_id in other_users:
            hist = data_final.loc[
                data_final["user_id"] == user_id, "month_payments_sum"
            ]

            median_val = hist.median() if len(hist) > 0 else 0

            median_rows.append({
                "user_id": user_id,
                "date": forecast_date,
                "amount": round(float(median_val), 2),
                "info": "median"
            })

        median_forecast_df = pd.DataFrame(median_rows)

        # собираем датафрейм
        if not median_forecast_df.empty:
            forecast_full_df = pd.concat(
                [forecast_final, median_forecast_df],
                axis=0,
                ignore_index=True
            )
        else:
            forecast_full_df = forecast_final.copy()

        forecast_full_df = forecast_full_df.sort_values(by=["date", "user_id"]).reset_index(drop=True)

        #forecast_full_df.to_csv(os.path.join(DATA_DIR, "forecast_all_next_month.csv"), index=False)
        path = os.path.join(DATA_DIR, "forecast_all_next_month.csv")
        forecast_full_df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        logger.info("ℹ️ Итоговый датафрейм прогнозов сформирован")

        # %%
        # генерируем запрос на запись прогнозов в БД
        payload = {
            "items": [
                {
                    "user_id": int(row.user_id),
                    "date": row.date,
                    "amount": float(row.amount),
                    "info": row.info
                }
                for row in forecast_full_df.itertuples(index=False)
            ]
        }
        
        # %%
        # отправляем запрос на запись в БД
        response = requests.post(API_URL_UP, json=payload, headers=headers_up, timeout=30)

        if response.status_code != 200:
            logger.info(f"❌ Ошибка записи в БД: {response.status_code}, {response.text}")
            raise RuntimeError("❌ Ошибка записи в БД")

        logger.info(f"Статус загрузки в БД: {response.status_code}, {response.text}")

    # %%
    logger.info(f"✅ Завершение скрипта")