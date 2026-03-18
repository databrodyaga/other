# %%
#!/usr/bin/env python3

# Прогнозирование платежей: SARIMAX и CatBoost (Месячная агрегация)
# Упрощенная версия: Без валидации и ансамбля.

# %%
import sys
import subprocess

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
    if spacy.__version__ != "3.6.1":
        subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy==3.6.1", "--force-reinstall"], stdout=subprocess.DEVNULL)
        import importlib
        importlib.reload(spacy)
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy==3.6.1", "--force-reinstall"], stdout=subprocess.DEVNULL)
    import spacy

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "https://github.com/explosion/spacy-models/releases/download/ru_core_news_sm-3.6.0/ru_core_news_sm-3.6.0-py3-none-any.whl"],
    stdout=subprocess.DEVNULL)

# %%
# 📦 стандартные библиотеки Python
import os
import random
import re
import multiprocessing
import logging

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

# 📈 временные ряды
import statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

# создаем функцию скользящего среднего (АДАПТИРОВАНА ПОД МЕСЯЦЫ)
def rolling_mean(group, rolling_window, lags):
    group = group.sort_values("date")
    
    # Скользящее среднее (квартальное, например, если window=3)
    group["rolling_month_payments"] = (
        group["month_payments_sum"]
        .shift(1)
        .rolling(window=rolling_window, min_periods=1)
        .mean()
    )
    
    # Лаги (по месяцам: 1, 2, 3, 6, 12)
    for lag in lags:
        group[f"month_payments_sum_lag_{lag}"] = group["month_payments_sum"].shift(lag)

    return group

# создаем функцию заполнения пропусков нулями в заданном временном ряду (АДАПТИРОВАНА ПОД МЕСЯЦЫ)
def fill_missing_dates(group):
    current_user_id = group.name

    end_date = pd.Timestamp.today().replace(day=1) - pd.DateOffset(days=1)
    end_date = end_date.replace(day=1) # Снова к первому числу
    
    if group['date'].max() > end_date:
        end_date = group['date'].max()

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
    
    # Проверка на достаточность данных для сезонности
    if len(y_train) <= seasonality:
        scale = np.mean(np.abs(np.diff(y_train))) # Если данных мало, берем лаг 1
    else:
        scale = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
        
    if scale == 0:
        # Fallback если сезонная разница равна 0
        scale = np.mean(np.abs(np.diff(y_train))) + 1e-8
        
    if scale == 0:
         logger.error("❌ Деление на ноль в MASE даже после fallback!")
         return np.inf

    return np.mean(np.abs(y_true - y_pred)) / scale

# создаем функцию проверки начальных значений на нули или слишком разреженные данные
def start_date_define(df, target_col, window_size=6, min_nonzero=3, num_windows=2):
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
# настраиваем логгер
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# файл
file_handler = logging.FileHandler("monthly_log.log")
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter(
    "%(asctime)s - %(levelname)-8s - %(funcName)-20s: %(lineno)-4d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_format)

# консоль
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
console_handler.setFormatter(console_format)

# добавляем обработчики в логгер
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Игнорируем ворнинги для малых выборок
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# %%

logger.info(f"✅ Запуск скрипта (Месячная агрегация - 2 модели)")

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
f"{'':21}statsmodels     {statsmodels.__version__ if statsmodels else 'not installed'}",
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

# устанавливаем значение для случайных значений и другие константы
RANDOM_STATE = 42
ROLLING_WINDOW = 3 
LAGS = [1, 2, 3, 6, 12] 

# %%
# импортируем данные
path1 = "3try_data_with_purpose_mean.parquet"

if os.path.exists(path1):
    data = pd.read_parquet(path1)
    logger.info(f"ℹ️ Данные загружены из файла: {path1}")

else:
    logger.info('ℹ️ Файл с данными не обнаружен. Начинаем загрузку данных с сервера...')
    import requests

    url = "https://api.lemonpie.tech/api/payments/ai"
    headers = {"Authorization": "Bearer YOUR_API_TOKEN"}

    params = {
        "limit": 5000,
        "page": 1,
        "expenditure": "incoming"
    }

    all_data = []

    with tqdm(desc="Загружено страниц", unit=" стр", dynamic_ncols=True) as pbar:
        while True:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code != 200:
                logger.error(f"❌ Ошибка загрузки данных с сервера: {response.status_code}")
                break

            result = response.json()
            page_data = result.get("data", [])
            if not page_data:
                break

            all_data.extend(page_data)

            params["page"] += 1
            pbar.update(1)

    data_full = pd.json_normalize(all_data, sep="__")
    logger.info(f"ℹ️ Данные загружены с сервера. Количество записей: {len(data_full)}")

    # заполняем accounts__user_id значениями из других столбцов
    data_full["accounts__user_id"] = (
        data_full["accounts__user_id"]
        .fillna(data_full["articles__user_id"])
        .fillna(data_full["projects__user_id"])
        .fillna(data_full["counterparties__user_id"])
        .fillna(data_full["robots__user_id"])
        .fillna(data_full["article_alternative_names__user_id"])
    )

    # отфильтруем данные
    data_actual_id_wodepo = data_full[
        ~data_full["purpose"].str.contains("вклад", na=False)
        & ~data_full["purpose"].str.contains("депози", na=False)
        & ~data_full["purpose"].str.contains("собствен", na=False)
        & ~data_full["purpose"].str.contains("процент", na=False)
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
    logger.info(f'ℹ️ Тексты закодированы и сохранены в файл {path1}')

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

data_fe["month_date"] = data_fe["date"].astype('datetime64[M]')

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

## сохраним/обновим фактические данные
path6 = "actual_monthly_payments.csv"
if os.path.exists(path6):
    actuals = pd.read_csv(path6, index_col=0, parse_dates=True)
    if data_final.index.max() > actuals.index.max():    
        actuals_new = data_final[data_final.index > actuals.index.max()][['user_id', 'month_payments_sum']]
        actuals_new.rename(columns={'user_id': 'fund_id'}, inplace=True)
        actuals_added = pd.concat([actuals, actuals_new], axis=0)
    else:
        actuals_added = actuals
else:
    actuals_added = data_final[['user_id', 'month_payments_sum']]
    actuals_added.rename(columns={'user_id': 'fund_id'}, inplace=True)

actuals_added = (
    actuals_added
    .reset_index()
    .sort_values(by=['index', 'fund_id'], kind='stable')
    .set_index('index')
)
actuals_added.to_csv(path6)
##

logger.info('ℹ️ Данные обработаны и подготовлены к обучению')
data_final.to_csv('data_monthly_payments.csv')

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
    .apply(lambda x: (x.iloc[-3:]['month_payments_sum'] != 0).sum())
)

# фильрация
funds = months_per_user_id[
    (months_per_user_id >= 18) &
    (months_per_user_nonzero > 0)
].index.tolist()

logger.info(f'ℹ️ Фильтрация по количеству активных месяцев и наличию платежей проведена, отобраны следующие фонды: {funds}')

# %%
# запускаем общий цикл обучения
logger.info('ℹ️ Запущен общий цикл обучения и прогнозирования по списку отобранных фондов')

mode = "train_val"  # production/train_val
logger.info(f"ℹ️ Режим прогнозирования: {mode}")

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

    # Оставляем только полные месяцы (все, что строго меньше начала текущего месяца)
    current_month_start = pd.Timestamp.today().replace(day=1)
    
    # Если в данных есть даты больше или равные текущему месяцу — отрезаем их
    data_fund = data_fund[data_fund.index < current_month_start]

    # %%
    # создаём версию со сдвигом для train/val
    data_fund_shifted = data_fund.copy()
    data_fund_shifted["target_shifted"] = data_fund_shifted["month_payments_sum"].shift(-1)
    data_fund_shifted = data_fund_shifted[:-1]  # убираем последнюю строку с NaN

    # Разделение на TRAIN и TEST (без валидации)
    if mode == "train_val":
        # Последний месяц для теста
        #test_start_date = data_fund_shifted.index.max() 
        #train_data_fund = data_fund_shifted.loc[: test_start_date - pd.#DateOffset(months=1)]
        #test_data_fund = data_fund_shifted.loc[test_start_date:]

        # тест 2 месяца
        test_start_date = data_fund_shifted.index.max()
        train_data_fund = data_fund_shifted.loc[: test_start_date - pd.DateOffset(months=1)]
        test_data_fund = data_fund_shifted.loc[test_start_date:]


    elif mode == "production":
        test_start_date = data_fund.index.max() # Текущий последний месяц
        
        # Train = всё доступное с историей
        train_data_fund = data_fund_shifted.loc[: test_start_date - pd.DateOffset(months=1)]
        test_data_fund = data_fund.loc[[test_start_date]]

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
    # SARIMAX
    # подготовка данных. загружаем ряд с несдвинутым целевым признаком

    if mode == "train_val":
        # ИСПРАВЛЕНИЕ:
        # 1. Обучаем SARIMAX на всех данных ВКЛЮЧАЯ текущий месяц (<= test_start_date), 
        #    чтобы прогноз (steps=1) был на СЛЕДУЮЩИЙ месяц (как у CatBoost).
        y_train_sarimax = data_fund.loc[data_fund.index <= test_start_date, "month_payments_sum"].copy()
        
        # 2. В тест берем строго всё, что ПОСЛЕ test_start_date (это и есть наш таргет)
        #    и ограничиваем iloc[:1], чтобы взять ровно один месяц
        y_test_sarimax = data_fund.loc[data_fund.index > test_start_date, "month_payments_sum"].iloc[:1].copy()

    elif mode == "production":
        y_train_sarimax = data_fund.loc[data_fund.index <= test_start_date, "month_payments_sum"].copy()
        # Для продакшена y_test_sarimax не существует (это будущее)

    logger.debug(f'🔍 Размеры SARIMAX: train={y_train_sarimax.shape}, test_target={len(y_test_sarimax) if mode=="train_val" else 0}')

    # %%
    # Построение и обучение модели SARIMAX
    logger.info("ℹ️ Обучение SARIMAX и прогнозирование начаты")

    # Функция кросс-валидаци для подбора гиперпараметров (на Train)
    n_splits = 3
    def cross_val_sarima(data, order, seasonal_order, n_splits=n_splits):
        if len(data) < n_splits + 5: # Если совсем мало данных
             return np.inf, np.inf, None
             
        tscv = TimeSeriesSplit(n_splits=n_splits)
        rmse_scores = []
        best_model_res = None
        best_rmse_fold = np.inf

        data = data.asfreq("MS") 

        for fold, (train_index, test_index) in enumerate(tscv.split(data)):
            train_fold = data.iloc[train_index]
            test_fold = data.iloc[test_index]
            
            if len(train_fold) < 13 or len(np.unique(train_fold)) == 1:
                 continue

            try:
                model_sarimax = SARIMAX(
                    train_fold,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False, 
                    enforce_invertibility=False,
                )
                results = model_sarimax.fit(disp=False, maxiter=200, method="powell")
                forecast = results.get_forecast(steps=len(test_fold))
                predicted_mean = forecast.predicted_mean.clip(lower=0)

                if len(test_fold) > 0:
                    current_rmse = np.sqrt(mean_squared_error(test_fold, predicted_mean))
                    rmse_scores.append(current_rmse)
                    
                    if current_rmse < best_rmse_fold:
                        best_rmse_fold = current_rmse
                        best_model_res = results # Сохраняем просто как флаг успеха
            except:
                continue

        if not rmse_scores:
            return np.inf, np.inf, None
            
        return np.mean(rmse_scores), 0, best_model_res

    # задаем наборы гиперпараметров для SARIMAX
    param_sets = [
        ((1, 1, 0), (0, 1, 1, 12)), 
        ((0, 1, 1), (0, 1, 1, 12)),
        ((1, 0, 0), (1, 1, 0, 12)),
        ((1, 1, 0), (0, 0, 0, 0)), 
    ]

    best_order = (1, 1, 0)
    best_seasonal_order = (0, 1, 1, 12)
    best_rmse_overall_sarimax = np.inf

    # Подбор параметров на Train
    for param in param_sets:
        avg_rmse, _, _ = cross_val_sarima(y_train_sarimax, param[0], param[1])
        if avg_rmse < best_rmse_overall_sarimax:
            best_rmse_overall_sarimax = avg_rmse
            best_order = param[0]
            best_seasonal_order = param[1]

    # Финальное обучение на всем Train
    try:
        final_model_sarimax = SARIMAX(
            y_train_sarimax.asfreq("MS"),
            order=best_order,
            seasonal_order=best_seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        final_res_sarimax = final_model_sarimax.fit(disp=False, maxiter=500, method="powell")
        
        # Прогноз на 1 шаг (Test)
        sarima_pred_obj = final_res_sarimax.get_forecast(steps=1)
        y_test_pred_sarimax_val = max(0, sarima_pred_obj.predicted_mean.item())
        
    except Exception as e:
        logger.error(f"SARIMAX Fit Error: {e}")
        y_test_pred_sarimax_val = 0.0

    # SARIMAX Метрики (если есть y_test)
    if mode == "train_val":
        # y_test_sarimax у нас уже есть
        # Приводим прогноз к Series с индексом
        y_test_pred_sarimax_series = pd.Series([y_test_pred_sarimax_val], index=y_test_sarimax.index)
        
        rmse_test_sarimax = np.sqrt(mean_squared_error(y_test_sarimax, y_test_pred_sarimax_series))
        mase_test_sarimax = mase(y_test_sarimax, y_test_pred_sarimax_series, y_train_sarimax, 12)
    
    logger.info("ℹ️ Обучение SARIMAX и прогнозирование завершены")

    # %%
    # Catboost
    logger.info("ℹ️ Обучение Catboost и прогнозирование начаты")

    # Проверка фолдов
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
        "model__iterations": [100, 300],
        "model__depth": [2, 4],
        "model__learning_rate": [0.05, 0.01],
        "model__l2_leaf_reg": [3],
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

    # Прогноз на Test
    y_test_pred_cb_val = np.clip(final_model_cb.predict(X_test), 0, None)
    if isinstance(y_test_pred_cb_val, np.ndarray):
        y_test_pred_cb_val = y_test_pred_cb_val[0]

    # CatBoost Метрики
    if mode == "train_val":
        rmse_test_cb = np.sqrt(mean_squared_error(y_test, [y_test_pred_cb_val]))
        mase_test_cb = mase(y_test, [y_test_pred_cb_val], y_train, 12)

    logger.info("ℹ️ Обучение Catboost и прогнозирование завершены")
    
    # %%
    # Сохранение результатов (БЕЗ АНСАМБЛЯ)
    if mode == "train_val":
        # Сохраняем метрики двух моделей
        path3 = "comparison_train_val_monthly.csv"
        if os.path.exists(path3):
            comparison = pd.read_csv(path3)
        else:
            comparison = pd.DataFrame(columns=["fund_id"])

        if FUND_ID in comparison["fund_id"].values:
            idx = comparison[comparison["fund_id"] == FUND_ID].index
            comparison.loc[idx, "sarimax_rmse"] = rmse_test_sarimax
            comparison.loc[idx, "sarimax_mase"] = mase_test_sarimax
            comparison.loc[idx, "catboost_rmse"] = rmse_test_cb
            comparison.loc[idx, "catboost_mase"] = mase_test_cb
        else:
            new_row = pd.DataFrame({
                "fund_id": [FUND_ID],
                "sarimax_rmse": [rmse_test_sarimax],
                "sarimax_mase": [mase_test_sarimax],
                "catboost_rmse": [rmse_test_cb],
                "catboost_mase": [mase_test_cb],
            })
            comparison = pd.concat([comparison, new_row], ignore_index=True)
        comparison.to_csv(path3, index=False)

    elif mode == "production":
        # Сохраняем прогнозы
        predict_date = data_fund.index[-1] + pd.DateOffset(months=1)
        
        forecasts_test = pd.DataFrame({
            "fund_id": FUND_ID,
            "forecast_sarimax": [y_test_pred_sarimax_val],
            "forecast_catboost": [y_test_pred_cb_val]
        }, index=[predict_date])

        path4 = "forecasts_full_monthly.csv"
        file_exists = os.path.isfile(path4)
        forecasts_test.to_csv(path4, mode="a", header=not file_exists, index=True)

# %%
# Усреднение для production (Простое среднее, т.к. нет валидации для весов)
if mode == 'production':
    if 'predict_date' in locals():
        forecast_date = predict_date.strftime('%Y-%m-%d')
        
        if os.path.exists(path4):
            forecasts_df = pd.read_csv(path4, index_col=0)
            # Берем только нужную дату
            forecast_next = forecasts_df[forecasts_df.index == forecast_date].copy()
            
            # Простое среднее двух моделей
            forecast_next['forecast'] = (forecast_next['forecast_sarimax'] + forecast_next['forecast_catboost']) / 2
            forecast_next['forecast'] = forecast_next['forecast'].round(2)
            
            forecast_final = forecast_next[['fund_id', 'forecast']]

            # Медианный прогноз для остальных (кого отсеяли)
            other_funds = list(set(data_final['user_id'].unique()) - set(forecast_final['fund_id'].unique()))
            median_rows = []

            for o_fund in other_funds:
                # Медиана по истории
                hist = data_final[data_final['user_id'] == o_fund]['month_payments_sum']
                if len(hist) > 0:
                    median_val = hist.median()
                else:
                    median_val = 0
                median_rows.append({
                    'fund_id': o_fund,
                    'forecast': round(median_val, 2)
                })
            
            median_forecast_df = pd.DataFrame(median_rows)
            if not median_forecast_df.empty:
                median_forecast_df.index = [forecast_date] * len(median_forecast_df)
                forecast_full = pd.concat([forecast_final, median_forecast_df], axis=0)
            else:
                forecast_full = forecast_final

            forecast_full = forecast_full.reset_index().sort_values(by=['index', 'fund_id']).set_index('index')
            forecast_full.index.name = None

            path5 = 'forecast_next_month.csv'
            if os.path.exists(path5):
                df_old = pd.read_csv(path5, index_col=0)
                df = pd.concat([df_old, forecast_full], axis=0)
            else:
                df = forecast_full

            df.to_csv(path5)
            logger.info(f'ℹ️ Итоговые прогнозы (Average Model) сохранены в файл {path5}')

logger.info(f"✅ Завершение скрипта")