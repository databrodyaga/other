#!/usr/bin/env python3

# Прогнозирование платежей с применением ансамбля моделей на реальных данных

# Прогнозируем суммы платежей на агрегированных данных на следующий день c учетом временной составляющей с применением трех моделей: LSTM, SARIMAX, Catboost и последующим взвешиванием прогнозов с помощью нейронной сети.

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

        # Переносим каждый тензор на устройство
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

# создаем функцию скользящего среднего
def rolling_mean(group, rolling_window, lags):
    group = group.sort_values("date")
    group["rolling_day_payments"] = (
        group["day_payments_sum"]
        .shift(1)
        .rolling(window=rolling_window, min_periods=1)
        .mean()
    )
    for lag in range(1, lags):
        group[f"day_payments_sum_lag_{lag}"] = group["day_payments_sum"].shift(lag)

    return group

# создаем функцию заполнения пропусков нулями в заданном временном ряду
def fill_missing_dates(group):
    current_user_id = group.name

    end_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    #end_date = pd.to_datetime('2025-07-25')
    full_range = pd.date_range(group['date'].min(), end_date, freq='D')
    #full_range = pd.date_range(group['date'].min(), group['date'].max(), freq='D')

    original_rows = len(group)
    new_rows = len(full_range)

    logger.debug(
        f"🔍 Заполнение дат для user_id='{current_user_id}'. "
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
def mase(y_true, y_pred, y_train, seasonality=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_train = np.asarray(y_train)
    scale = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
    if scale == 0:
        logger.error("❌ Деление на ноль в MASE!")
        logger.debug("🔍 seasonality =", seasonality)
        logger.debug("🔍 y_train:", y_train)
        logger.debug("🔍 y_true:", y_true)
        logger.debug("🔍 y_pred:", y_pred)
        logger.debug("🔍 Разность:", y_true - y_pred)
        logger.debug("🔍 y_train[seasonality:]:", y_train[seasonality:])
        logger.debug("🔍 y_train[:-seasonality]:", y_train[:-seasonality])
        sys.exit("Программа остановлена из-за деления на ноль в MASE.")

    return np.mean(np.abs(y_true - y_pred)) / scale

# создаем функцию проверки начальных значений на нули или слишком разреженные данные
# находим 7 временных окон по 30 дней подрядк, где больше 15 не нулевых значений и отбрасываем все, что до них
def start_date_define(df, target_col, window_size=30, min_nonzero=15, num_windows=7):
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
file_handler = logging.FileHandler("second_try_log.log")
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
    f"{'':21}statsmodels     {statsmodels.__version__ if statsmodels else 'not installed'}",
    f"{'':21}CUDA available  {torch.cuda.is_available()}"
    ]

    logger.info(f"🔍 Версии библиотек:\n" + "\n".join(versions))

    # зададим стандарт датафрейма перед загрузкой
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:.2f}".format)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", 
                            message="DataFrameGroupBy.apply operated on the grouping columns.*",
                            category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # устанавливаем значение для случайных значений и другие константы
    RANDOM_STATE = 42
    ROLLING_WINDOW = 7 # размер окна скользящего среднего
    LAGS = 8 # количество лагов

    # %%
    # импортируем данные
    path1 = "2try_data_with_purpose_mean.parquet"

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
            "expenditure": "incoming",
            #"date_from": "2025-11-24",
            #"date_to": "2025-11-25"   
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
                
        # преобразуем в таблицу (вложенные поля будут с __)
        data_full = pd.json_normalize(all_data, sep="__")
        logger.info(f"ℹ️ Данные загружены с сервера. Количество записей: {len(data_full)}")
        #data_full.to_parquet("data_download.parquet", index=False)

        # %%
        # заполняем accounts__user_id значениями из других столбцов
        data_full["accounts__user_id"] = (
            data_full["accounts__user_id"]
            .fillna(data_full["articles__user_id"])
            .fillna(data_full["projects__user_id"])
            .fillna(data_full["counterparties__user_id"])
            .fillna(data_full["robots__user_id"])
            .fillna(data_full["article_alternative_names__user_id"])
        )
        
        # отфильтруем данные по актуальным пользователям и уберем технические поступления - возврат депозитов, переводы собственных средств
        data_actual_id_wodepo = data_full[
            ~data_full["purpose"].str.contains("вклад", na=False)
            & ~data_full["purpose"].str.contains("депози", na=False)
            & ~data_full["purpose"].str.contains("собствен", na=False)
            & ~data_full["purpose"].str.contains("процент", na=False)
        ]

        # %%
        # удаляем столбцы с дублирующими id, которые использовались при объединении данных при выгрузке из БД, также пустые столбцы и техстолбцы с одним значением
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

        # %%
        # поправим типы данных и заполним пропуски метками missing (для текстовых значений категорий) и 0 для пропущенных ID
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

        # конвертируем дату в datetime
        data["date"] = pd.to_datetime(data["date"], yearfirst=True, errors='coerce')

        # и убираем записи из будущего и меньше нуля (и такое бывает)
        yesterday = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
        data = data[data["date"] <= yesterday]
        data = data[data["payments_amount"] > 0]

        # переименуем и поправим тип столбца с фондами
        data = data.rename(columns={"accounts__user_id": "user_id"})
        data["user_id"] = data["user_id"].fillna(0).astype("int64")

        # %%
        # закодируем текстовое поле
        logger.info("ℹ️ Закодируем текстовое поле")
            
        # сначала очищаем и лемматизируем тексты
        data["clean_purpose"] = preprocess_texts_optimized(texts=data["purpose"],nlp_model_name="ru_core_news_sm")

        # грузим модели 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")
        model = model.to(device)
        
        # и запускаем генерацию эмбеддингов
        data["purpose_emb"] = get_embeddings_batch(data["clean_purpose"], tokenizer, model, device)
        
        # 1. усредняем эмбеддинг в одно число
        data["purpose_mean"] = data["purpose_emb"].apply(lambda x: float(np.mean(x)))
        
        # 2. выделяем три главные компоненты с предварительным масштабированием и по батчам  
        batch_size = 10_000
        scaler = StandardScaler()
        ipca = IncrementalPCA(n_components=3)

        # обучаем скейлер
        for i in tqdm(range(0, len(data), batch_size), desc="Обучение StandardScaler"):
            batch = np.vstack(data["purpose_emb"].iloc[i:i+batch_size])
            scaler.partial_fit(batch)

        # обучаем PCA на масштабированных данных
        for i in tqdm(range(0, len(data), batch_size), desc="Обучение IncrementalPCA"):
            batch = np.vstack(data["purpose_emb"].iloc[i:i+batch_size])
            batch_scaled = scaler.transform(batch)
            ipca.partial_fit(batch_scaled)

        # применяем трансформацию ко всему массиву
        transformed_batches = []
        for i in tqdm(range(0, len(data), batch_size), desc="Масштабируем эмбеддинги"):
            batch = np.vstack(data["purpose_emb"].iloc[i:i+batch_size]).astype(np.float32)
            batch_scaled = scaler.transform(batch)
            transformed_batches.append(ipca.transform(batch_scaled))
            
        purpose_pca_features = np.vstack(transformed_batches)

        # делаем датафрейм
        pca_column_names = [f"purpose_pca_{i+1}" for i in range(3)]
        data[pca_column_names] = purpose_pca_features

        # удалим ненужные столбцы
        data.drop(columns=["purpose", "clean_purpose", "purpose_emb"], inplace=True)
        
        # сохраним результат
        data.to_parquet(path1, index=False)
        logger.info(f'ℹ️ Тексты закодированы и сохранены в файл {path1}')

    # %%
    data = data.drop("id", axis=1)  # удаляем id операции перед агрегированием

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

    # создаем временный столбец с общим количеством записей для каждого пользователя
    data_fe["user_total_counts_col"] = data_fe.groupby("user_id")["user_id"].transform("size")

    # кодируем каждый из столбцов id_cols
    for col in id_cols:

        encoded_col_name = f"{col}_ufreq"  # задаем имя столбцу
        # считаем количество вхождений категории 'col' для данного 'user_id'
        numerator = data_fe.groupby(["user_id", col])[col].transform("size")
        # считаем долю и записываем в столбец
        data_fe[encoded_col_name] = numerator.astype(float) / data_fe["user_total_counts_col"].astype(float)

    # удаляем временный столбец с общим количеством
    data_fe.drop(columns=["user_total_counts_col"], inplace=True)

    # %%
    # агрегация платежей по дням и пользователям
    logger.info('ℹ️ Агрегируем атомарные данные до уровня дней')
    # делаем список колонок для расчета nunique
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

    # делаем список колонок с кодированными значениями текстового поля
    purpose_mean_cols = ["purpose_mean", "purpose_pca_1", "purpose_pca_2", "purpose_pca_3"]

    # делаем список кодированных категориальных колонок
    col_ufreq = [col for col in data_fe.columns if col.endswith("_ufreq")]

    # агрегируем данные до уровня дня
    data_daily_agg = (
        data_fe.groupby(["user_id", "date"])
        .agg(
            **{f"day_{col}_nunique": (col, "nunique") for col in nunique_cols},
            day_payments_sum=("payments_amount", "sum"),
            day_payments_count=("payments_amount", "count"),
            day_min_payment=("payments_amount", "min"),
            day_max_payment=("payments_amount", "max"),
            **{f"day_{col}": (col, "mean") for col in purpose_mean_cols},
            **{f"{col}_mean": (col, "mean") for col in col_ufreq},
            **{f"{col}_std": (col, "std") for col in col_ufreq},
        )
        .reset_index()
    )

    # заполним NaN в std колонках нулями (там где было одно наблюдение/платеж)
    std_cols = [col for col in data_daily_agg.columns if col.endswith("_std")]
    data_daily_agg[std_cols] = data_daily_agg[std_cols].fillna(0)
    # и округлим
    for col in data_daily_agg.columns:
        if col.endswith("_ufreq_mean") or col.endswith("_ufreq_std"):
            data_daily_agg[col] = data_daily_agg[col].round(6)
        elif col in ["day_payments_sum", "day_min_payment", "day_max_payment"]:
            data_daily_agg[col] = data_daily_agg[col].round(2)

    # %%
    # заполним пропущенные дни нулевыми значениями, чтобы сохранить структуру временного ряда
    logger.info('ℹ️ Заполняем пропуски в агрегированных данных (дневной временной ряд) нулями')
    data_daily_agg_filled = data_daily_agg.groupby("user_id", group_keys=False).apply(fill_missing_dates)

    # добавляем столбцы с днем недели и номером недели в месяце + кодируем их с помощью циклического кодирования (sin/cos)
    logger.info('ℹ️ Добавляем дни недели и номера недели в месяце')
    # день недели (0 — понедельник, 6 — воскресенье)
    data_daily_agg_filled["day_of_week"] = data_daily_agg_filled["date"].dt.weekday
    data_daily_agg_filled["day_of_week_sin"] = np.sin(2 * np.pi * data_daily_agg_filled["day_of_week"] / 7).round(6)
    data_daily_agg_filled["day_of_week_cos"] = np.cos(2 * np.pi * data_daily_agg_filled["day_of_week"] / 7).round(6)

    # номер недели в месяце (1–5, 6 неделя скорее добавит шума,и вряд ли выучится хорошо)
    data_daily_agg_filled["week_number"] = (data_daily_agg_filled["date"].dt.day.sub(1).floordiv(7).add(1))
    data_daily_agg_filled["week_number_sin"] = np.sin(2 * np.pi * data_daily_agg_filled["week_number"] / 5).round(6)
    data_daily_agg_filled["week_number_cos"] = np.cos(2 * np.pi * data_daily_agg_filled["week_number"] / 5).round(6)

    # %%
    # добавим по платежам скользящее среднее  и лаги за последние LAGS дней, исключая текущий день
    logger.info('ℹ️ Добавляем скользящие средние и лаги по целевому признаку')

    # применим функцию скользящего среднего к каждой группе после группировки по user_id
    data_daily_agg_filled_added = data_daily_agg_filled.groupby(
        "user_id", group_keys=False
    ).apply(lambda group: rolling_mean(group, rolling_window=ROLLING_WINDOW, lags=LAGS))

    data_daily_agg_filled_added = data_daily_agg_filled_added.fillna(0)

    # %%
    # преобразуем дату в индекс
    data_final = data_daily_agg_filled_added.set_index("date", drop=True)
    data_final.index.name = None

    # %%
    # удаляем лишние колонки
    cols_to_drop_6 = [
        "day_robots__id_nunique",
        "day_counterpartie_id_nunique",
        "day_projects__parent_id_nunique",
        "day_donor_id_nunique",
        "day_articles__parent_id_nunique",
        "day_counterparties__parent_id_nunique",
        "day_account_id_nunique",
        "day_article_id_nunique",
        "day_project_id_nunique",
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

    ## сохраним/обновим фактические данные для дальнейшего анализа и расчета метрик
    path6 = "actual_day_payments.csv"
    if os.path.exists(path6):
        actuals = pd.read_csv(path6, index_col=0, parse_dates=True)
        if data_final.index.max() > actuals.index.max():    
            actuals_new = data_final[data_final.index > actuals.index.max()][['user_id', 'day_payments_sum']]
            actuals_new.rename(columns={'user_id': 'fund_id'}, inplace=True)
            actuals_added = pd.concat([actuals, actuals_new], axis=0)
        else:
            actuals_added = actuals
    else:
        actuals_added = data_final[data_final.index > '2025-07-13'][['user_id', 'day_payments_sum']]
        actuals_added.rename(columns={'user_id': 'fund_id'}, inplace=True)

    # сортировка и сохранение фактических данных
    actuals_added = (
        actuals_added
        .reset_index()
        .sort_values(by=['index', 'fund_id'], kind='stable')
        .set_index('index')
    )
    actuals_added.to_csv(path6)
    ##

    logger.info('ℹ️ Данные обработаны и подготовлены к обучению')

    # %%
    # запускаем обучение моделей
    # отберем для обучения фонды, для которых количество дней с данными больше заданного порога 
    # и в которых нет нулевого последнего месяца

    data_final = data_final.sort_index()

    # подсчёт количества активных дней
    days_per_user_id = data_final[data_final["day_payments_sum"] != 0].groupby("user_id")['day_payments_sum'].count()

    # проверка последих 30 дней на наличие данных
    days_per_user_nonzero_days = (
        data_final
        .sort_index()
        .groupby('user_id')
        .apply(lambda x: (x.iloc[-30:]['day_payments_sum'] != 0).sum())
    )

    # фильрация по количеству активных дней (по тестам определен порог в 150 дней) и проверке последних 30 дней
    funds = days_per_user_id[
        (days_per_user_id >= 150) &
        (days_per_user_nonzero_days > 0)
    ].index.tolist()

    logger.info(f'ℹ️ Фильтрация по количеству активных дней и наличию платежей в текущем месяце проведена, отобраны следующие фонды: {funds}')

    #funds = [176,784,791,799,886,1001] #237

    # %%
    # запускаем общий цикл обучения и прогнозирования по списку отобранных фондов
    logger.info('ℹ️ Запущен общий цикл обучения и прогнозирования по списку отобранных фондов')

    mode = "production"  # production/train_val
    logger.info(f"ℹ️ Режим прогнозирования: {mode}")

    for fund in tqdm(funds):

        # %%
        # выберем фонд
        FUND_ID = fund
        logger.info(f"🔹 FUND_ID: {FUND_ID}")

        # устанавливаем сиды - для каждого фонда свой(для torch)
        user_specific_seed = RANDOM_STATE + int(FUND_ID) 
        seed_all(user_specific_seed)

        data_fund = data_final[data_final["user_id"] == FUND_ID].copy()  # 185

        # проверяем монотонность и сортируем при необходимости
        if not data_fund.index.is_monotonic_increasing:
            data_fund = data_fund.sort_index()

        #проверяем ряд на стартовые нули и разреженные значения
        data_fund = start_date_define(data_fund, target_col="day_payments_sum")

        if data_fund is None:
            logger.warning(f"⚠️ Фонд {FUND_ID} пропускаем — ряд данных слишком разреженный")
            continue
        
        # убираем user_id
        data_fund.drop(["user_id"], axis=1, inplace=True)

        # %%
        # заменяем агрегированные дневные суммы из THRESHOLD_CONST перцентиля на медианное значение
        TRESHOLD_CONST = 0.90

        threshold = data_fund["day_payments_sum"].quantile(TRESHOLD_CONST)
        data_fund.loc[data_fund["day_payments_sum"] > threshold, "day_payments_sum"] = data_fund["day_payments_sum"].median()

        # %%

        # создаём версию со сдвигом для train/val
        data_fund_shifted = data_fund.copy()
        data_fund_shifted["target_shifted"] = data_fund_shifted["day_payments_sum"].shift(
            -1
        )
        data_fund_shifted = data_fund_shifted[:-1]  # убираем последнюю строку с NaN
        
        # делим данные на тренировочные, валидационные и тестовые
        if mode == "train_val":
            test_start_date = data_fund_shifted.index.max().replace(day=1)
            val_start_date = (test_start_date - pd.DateOffset(months=1)).replace(day=1)
            val_end_date = test_start_date - pd.Timedelta(days=1)

            # делим данные
            train_data_fund = data_fund_shifted.loc[: val_start_date - pd.Timedelta(days=1)]
            val_data_fund = data_fund_shifted.loc[val_start_date:val_end_date]
            test_data_fund = data_fund_shifted.loc[test_start_date:]

        elif mode == "production":
            test_start_date = data_fund.index.max()
            val_start_date = test_start_date - pd.Timedelta(days=30)

            # делим данные
            train_data_fund = data_fund_shifted.loc[: val_start_date - pd.Timedelta(days=1)]
            val_data_fund = data_fund_shifted.loc[
                val_start_date : test_start_date - pd.Timedelta(days=1)
            ]
            test_data_fund = data_fund.loc[
                [test_start_date]
            ]  # одна строка без сдвига, для прогноза

        # проверяем фиксируем размеры и даты
        
        logger.debug(f'🔍 Размеры тренировочных, валидационных и тестовых данных для LSTM/Catboost: train={train_data_fund.shape}, val={val_data_fund.shape}, test={test_data_fund.shape}')
        logger.debug(f"🔍 Первая тренировочная дата для LSTM/Catboost: {train_data_fund.index.min().date()}")
        logger.debug(f"🔍 Последняя тренировочная дата для LSTM/Catboost: {train_data_fund.index.max().date()}")
        logger.debug(f"🔍 Первая валидационная дата для LSTM/Catboost: {val_data_fund.index.min().date()}")
        logger.debug(f"🔍 Последняя валидационная дата для LSTM/Catboost: {val_data_fund.index.max().date()}")
        logger.debug(f"🔍 Первая тестовая дата для LSTM/Catboost: {test_data_fund.index.min().date()}")
        logger.debug(f"🔍 Последняя тестовая дата для LSTM/Catboost: {test_data_fund.index.max().date()}")

        # %%
        # выделяем категоральные признаки
        cat_features = ["day_of_week", "week_number"]

        # формируем X_train и y_train
        X_train = train_data_fund.drop(["day_payments_sum", "target_shifted"], axis=1)
        y_train = train_data_fund["target_shifted"]

        # формируем X_val и y_val
        X_val = val_data_fund.drop(["day_payments_sum", "target_shifted"], axis=1)
        y_val = val_data_fund["target_shifted"]

        # формируем X_test и y_test
        if mode == "train_val":
            X_test = test_data_fund.drop(["day_payments_sum", "target_shifted"], axis=1)
            y_test = test_data_fund["target_shifted"]
        elif mode == "production":
            X_test = test_data_fund.drop(["day_payments_sum"], axis=1)

        # %%
        # LSTM
        # Подготовка данных

        # масштабируем числовые признаки
        scaler_x = StandardScaler()
        X_train_num_scaled = scaler_x.fit_transform(X_train.drop(columns=cat_features))
        X_val_num_scaled = scaler_x.transform(X_val.drop(columns=cat_features))
        X_test_num_scaled = scaler_x.transform(X_test.drop(columns=cat_features))

        # кодируем категориальные признаки
        X_train_cat_encoded = pd.get_dummies(X_train[cat_features], dtype=float)
        X_val_cat_encoded = pd.get_dummies(X_val[cat_features], dtype=float)
        X_test_cat_encoded = pd.get_dummies(X_test[cat_features], dtype=float)

        # Выравниваем столбцы (если в тестовой или валидационной выборке пропущены категории)
        X_train_cat_encoded, X_val_cat_encoded = X_train_cat_encoded.align(
            X_val_cat_encoded, join="left", axis=1, fill_value=0
        )
        _, X_test_cat_encoded = X_train_cat_encoded.align(
            X_test_cat_encoded, join="left", axis=1, fill_value=0
        )

        # объединяем обратно
        X_train_prepared = np.hstack([X_train_num_scaled, X_train_cat_encoded])
        X_val_prepared = np.hstack([X_val_num_scaled, X_val_cat_encoded])
        X_test_prepared = np.hstack([X_test_num_scaled, X_test_cat_encoded])

        # масштабируем целевую переменную
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1))
        if mode == "train_val":
            y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

        # преобразуем обратно в DataFrame для сохранения временных индексов
        X_train_prepared = pd.DataFrame(
            X_train_prepared,
            columns=list(X_train.drop(columns=cat_features).columns)
            + list(X_train_cat_encoded.columns),
            index=X_train.index,
        )

        X_val_prepared = pd.DataFrame(
            X_val_prepared,
            columns=list(X_val.drop(columns=cat_features).columns)
            + list(X_val_cat_encoded.columns),
            index=X_val.index,
        )

        X_test_prepared = pd.DataFrame(
            X_test_prepared,
            columns=list(X_test.drop(columns=cat_features).columns)
            + list(X_test_cat_encoded.columns),
            index=X_test.index,
        )

        # преобразуем в Series(см.выше)
        y_train_scaled = pd.Series(
            y_train_scaled.ravel(), name=y_train.name, index=y_train.index
        )
        y_val_scaled = pd.Series(y_val_scaled.ravel(), name=y_val.name, index=y_val.index)
        if mode == "train_val":
            y_test_scaled = pd.Series(
                y_test_scaled.ravel(), name=y_test.name, index=y_test.index
            )

        # %%
        # Построение и обучение модели
        logger.info("ℹ️ Обучение LSTM и прогнозирование начаты")
        # %%
        # задаем шаги и количество разбиений временного ряда
        time_steps = 7
        n_splits = 5

        # задаем устройство для обучения
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"ℹ️ Device: {device}")

        # функция создания последовательностей для LSTM
        def create_sequences(data, target, time_steps=1):
            X_seq, y_seq = [], []
            for i in range(len(data) - time_steps):
                X_seq.append(data.iloc[i : i + time_steps])
                y_seq.append(target.iloc[i + time_steps])
            return np.array(X_seq), np.array(y_seq)

        # формируем последовательности
        X_seq_full, y_seq_full = create_sequences(
            X_train_prepared, y_train_scaled, time_steps
        )

        # преобразуем в тензоры
        X_full_tensor = torch.tensor(X_seq_full, dtype=torch.float32).to(device)
        y_full_tensor = torch.tensor(y_seq_full, dtype=torch.float32).view(-1, 1).to(device)

        # фомируем модель LSTM
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(LSTMModel, self).__init__()
                self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
                # self.dropout1 = nn.Dropout(0.1)
                # self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
                # self.dropout2 = nn.Dropout(0.2)
                self.fc1 = nn.Linear(hidden_size, 256)
                self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
                self.fc2 = nn.Linear(256, output_size)

            def forward(self, x):
                x, _ = self.lstm1(x)
                # x = self.dropout1(x)
                # x, _ = self.lstm2(x)
                # x = self.dropout2(x)
                x = x[:, -1, :]
                x = self.fc1(x)
                x = self.leaky_relu(x)
                x = self.fc2(x)
                x = self.leaky_relu(x)
                return x

        # инициализация модели
        input_size = X_seq_full.shape[2]
        hidden_size = 256
        output_size = 1
        model_lstm = LSTMModel(input_size, hidden_size, output_size).to(device)

        # задаем функцию потерь и оптимизатор
        criterion = nn.HuberLoss(delta=1.5)
        optimizer = optim.AdamW(model_lstm.parameters(), lr=0.001, weight_decay=1e-4)

        # создаем ReduceLROnPlateau
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        # обучаем модель на всем тренировочном датасете
        num_epochs = 200
        for epoch in range(num_epochs):
            model_lstm.train()
            optimizer.zero_grad()
            output = model_lstm(X_full_tensor)
            loss = criterion(output, y_full_tensor)
            loss.backward()
            optimizer.step()

        #    scheduler.step(loss)

        # %%
        # добавляем последние time_steps строк из тренировочного набора в валидационный набор
        # (тк они нужны для запуска первого прогноза, чтобы не тратить на это валидационные данные)
        X_val_full = pd.concat([X_train_prepared.tail(time_steps), X_val_prepared], axis=0)
        y_val_full = pd.concat([y_train_scaled.tail(time_steps), y_val_scaled], axis=0)

        # создаем последовательности для валидационного набора
        X_seq_val, y_seq_val = create_sequences(X_val_full, y_val_full, time_steps)

        # преобразуем в тензоры
        X_val_tensor = torch.tensor(X_seq_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_seq_val, dtype=torch.float32).view(-1, 1)

        # делаем прогноз
        model_lstm.eval()
        with torch.no_grad():
            y_pred = model_lstm(X_val_tensor.to(device)).cpu().numpy()

        # разворачиваем масштабированные данные
        y_val_original = np.round(scaler_y.inverse_transform(y_val_tensor.numpy()).flatten(), decimals=2)
        y_val_pred_lstm = np.clip(np.round(scaler_y.inverse_transform(y_pred).flatten(), 2), 0, None)

        # преобразуем y_val_original в Series с правильными временныеми индексами
        y_val_original = pd.Series(y_val_original, index=y_val_full.index[-len(y_val_original) :])
        logger.info('ℹ️ Обучение LSTM и прогнозирование завершены')

        # %%
        # SARIMAX
        # подготовка данных. загружаем ряд с несдвинутым целевым признаком и разбиваем на тренировочную, валидационную и тестовую выборки

        if mode == "train_val":
            test_start_date = data_fund_shifted.index.max().replace(day=1)
            val_start_date = (test_start_date - pd.DateOffset(months=1)).replace(day=1)
            val_end_date = test_start_date - pd.Timedelta(days=1)
            # делим данные
            y_train_sarimax = data_fund.loc[data_fund.index < val_start_date, "day_payments_sum"].copy()
            y_val_sarimax = data_fund.loc[val_start_date:val_end_date, "day_payments_sum"].copy()
            y_test_sarimax = data_fund.loc[data_fund.index >= test_start_date, "day_payments_sum"].copy()

        elif mode == "production":
            test_start_date = data_fund.index.max()
            val_start_date = test_start_date - pd.Timedelta(days=30)

            # делим данные
            y_train_sarimax = data_fund.loc[data_fund.index < val_start_date, "day_payments_sum"].copy()
            y_val_sarimax = data_fund.loc[val_start_date : test_start_date - pd.Timedelta(days=1), "day_payments_sum"].copy()
            y_test_sarimax = data_fund.loc[[test_start_date], "day_payments_sum"].copy()

        # проверим границы наборов у Sarimax (данные не сдвинуты - поэтому должен был отличие от аналогичных выше на 1 день)
        logger.debug(f'🔍 Размеры тренировочных, валидационных и тестовых данных для SARIMAX: train={y_train_sarimax.shape}, val={y_val_sarimax.shape}, test={y_test_sarimax.shape}')
        logger.debug(f"🔍 Первая тренировочная дата для SARIMAX: {y_train_sarimax.index.min().date()}")
        logger.debug(f"🔍 Последняя тренировочная дата для SARIMAX: {y_train_sarimax.index.max().date()}")
        logger.debug(f"🔍 Первая валидационная дата для SARIMAX: {y_val_sarimax.index.min().date()}")
        logger.debug(f"🔍 Последняя валидационная дата для SARIMAX: {y_val_sarimax.index.max().date()}")
        logger.debug(f"🔍 Первая тестовая дата для SARIMAX: {y_test_sarimax.index.min().date()}")
        logger.debug(f"🔍 Последняя тестовая дата для SARIMAX: {y_test_sarimax.index.max().date()}")
        
        # %%
        # Построение и обучение модели
        logger.info("ℹ️ Обучение SARIMAX и прогнозирование начаты")

        # %%
        # функция кросс-валидаци для подбора гиперпараметров SARIMAX
        def cross_val_sarima(data, order, seasonal_order, n_splits=n_splits):
            tscv = TimeSeriesSplit(n_splits=n_splits)
            rmse_scores = []
            mase_scores = []
            best_model = None
            best_rmse = np.inf

            data = data.asfreq("D")

            for fold, (train_index, test_index) in enumerate(tscv.split(data)):
                train = data.iloc[train_index]
                test = data.iloc[test_index]
                
                # проверим фолды таргета на нули (так как могут быть многочисленные пропуски в данных) и если так, то пропускаем фолд
                if len(np.unique(train)) == 1 or len(np.unique(test)) == 1:
                    logger.warning(f"⚠️ Фолд {fold} пропускаем — целевой признак константный")
                    logger.warning(f'⚠️ Даты фолда: {train.index.min().date()} - {train.index.max().date()}')
                    continue

                model_sarimax = SARIMAX(
                    train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                )

                results = model_sarimax.fit(
                    disp=False,
                    maxiter=1000,
                    optim_score="harvey",
                    method="powell"
                )

                # делаем прогноз
                forecast = results.get_forecast(steps=len(test))
                predicted_mean = forecast.predicted_mean
                predicted_mean = predicted_mean.clip(lower=0)  # отрицательные меняем на ноль
                predicted_mean = predicted_mean.round(2)    

                # фильтруем пары без NaN
                mask = ~predicted_mean.isna() & ~test.isna()
                predicted_clean = predicted_mean[mask]
                test_clean = test[mask]

                # если после фильтрации остались данные — считаем метрики
                if len(test_clean) > 0:
                    # cчитаем RMSE
                    current_rmse = np.sqrt(mean_squared_error(test_clean, predicted_clean))
                    rmse_scores.append(current_rmse)
                    # cчитаем MASE
                    current_mase = mase(test_clean, predicted_clean, train)
                    mase_scores.append(current_mase)
                    
                    # сохраняем лучший вариант метрики и модели
                    if current_rmse < best_rmse:
                        best_rmse = current_rmse
                        best_model = results
                else:
                    logger.warning(f"⚠️ Фолд {fold} пропускаем — все значения NaN после прогноза")
                        
                
            # возвращаем средние значения RMSE и MASE
            return np.nanmean(rmse_scores), np.nanmean(mase_scores), best_model

        # задаем наборы гиперпараметров для SARIMAX
        param_sets = [
            ((1, 1, 0), (0, 1, 1, 7)),  # упрощенная модель
            ((0, 1, 1), (0, 1, 1, 7)),  # базовый вариант
            ((1, 0, 0), (1, 1, 0, 7)),  # AR-структура
            ((0, 1, 2), (0, 1, 1, 7)),  # с увеличенным MA
            ((0, 1, 1), (1, 1, 1, 7)),  # с сезонным AR
            ((1, 1, 7), (0, 0, 0, 0)),  # ручной вариант
            ((1, 1, 2), (0, 0, 0, 0)),  # ручной вариант 2
        ]

        # перебор параметров
        best_model_overall_sarimax = None
        best_rmse_overall_sarimax = np.inf
        best_mase_overall_sarimax = np.inf

        for param in param_sets:

            average_rmse_sarimax, average_mase_sarimax, best_model_sarimax = (
                cross_val_sarima(y_train_sarimax, param[0], param[1])
            )
            
            if average_rmse_sarimax < best_rmse_overall_sarimax:
                best_rmse_overall_sarimax = average_rmse_sarimax
                best_model_overall_sarimax = best_model_sarimax
                best_mase_overall_sarimax = average_mase_sarimax

        # %%
        # переобучаем лучшую модель на всем датасете инкрементально, чтобы использовать потом валидационные и тестовые прогнозы

        # забираем всю последовательность для обучения и прогноза
        y_full_sarimax = data_fund["day_payments_sum"].copy().asfreq("D")

        # создаем пустую серию для хранения прогнозов
        sarima_forecast = pd.Series(index=y_full_sarimax.index, dtype=float)

        # определяем минимальный размер выборки для начала обучения SARIMAX
        min_full_size = 30
        refit_interval = 7

        # заполняем начальные значения средними про предыдущим значениям
        sarima_forecast.iloc[0] = y_full_sarimax.iloc[0]  # первое значение как есть

        for i in range(1, min_full_size):
            sarima_forecast.iloc[i] = round(y_full_sarimax.iloc[:i].mean(), 2)

        # задаем параметры модели
        order = best_model_overall_sarimax.model.order
        seasonal_order = best_model_overall_sarimax.model.seasonal_order

        fit_params = {
            "disp": False,
            "maxiter": 1000,
            "optim_score": "harvey",
            "method": "powell"
        }
        
        logger.debug(f"🔍 Параметры SARIMAX: order={order}, seasonal_order={seasonal_order}")

        # запускаем обучение и прогнозирование, начиная с min_full_size
        for i in range(min_full_size, len(y_full_sarimax)):

            is_first_step = (i == min_full_size)
            is_refit_step = ((i - min_full_size) % refit_interval == 0)

            if is_first_step or is_refit_step:
                # полное переобучение
                # используем все доступные данные для обучения
                train_series = y_full_sarimax.iloc[:i]
                model_sarimax = SARIMAX(
                    train_series,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                sarimax_results = model_sarimax.fit(**fit_params)
            else:
                # быстрое обновление через .apply()
                new_observations = y_full_sarimax.iloc[i-1:i]
                
                # проверяем, чтобы не выбросило ошибку
                if 'description' not in sarimax_results.cov_kwds:
                    sarimax_results.cov_kwds['description'] = ''
                # применяем их к существующей модели
                sarimax_results = sarimax_results.apply(new_observations, refit=False)

            # прогнозируем следующее значение и заменяем отрицательные значения на 0
            forecast = sarimax_results.get_forecast(steps=1)
            raw_pred = forecast.predicted_mean.item()

            # если предсказание < 0 или > threshold — обнуляем как ошибочное
            if raw_pred < 0 or raw_pred > threshold:
                predicted_value = 0.0
            else:
                predicted_value = round(raw_pred, 2)
            
            sarima_forecast.iloc[i] = predicted_value

        # обнуляем первые значения, чтобы они не влияли на CatBoost
        sarima_forecast.iloc[:min_full_size+10] = np.nan

        # %%
        # валидация модели
        # отбираем прогнозы и реальные значения для валидации со сдвигом на 1 значение назад, для корректоного сравнения с другими моделями

        if mode == "train_val":

            # делим данные
            y_val_sarimax = (
                data_fund.shift(-1)
                .loc[val_start_date:val_end_date, "day_payments_sum"]
                .copy()
                .asfreq("D")
            )
            y_val_pred_sarimax = (
                sarima_forecast.shift(-1)
                .loc[val_start_date:val_end_date]
                .copy()
                .asfreq("D")
            )
            # выравнивание и отбрасывание NaN
            y_val_sarimax, y_val_pred_sarimax = y_val_sarimax.align(
                y_val_pred_sarimax, join="inner"
            )

        elif mode == "production":

            # делим данные
            y_val_sarimax = (
                data_fund.shift(-1)
                .loc[
                    val_start_date : test_start_date - pd.Timedelta(days=1),
                    "day_payments_sum",
                ]
                .copy()
                .asfreq("D")
            )
            y_val_pred_sarimax = (
                sarima_forecast.shift(-1)
                .loc[val_start_date : test_start_date - pd.Timedelta(days=1)]
                .copy()
                .asfreq("D")
            )
            # выравнивание и отбрасывание NaN
            y_val_sarimax, y_val_pred_sarimax = y_val_sarimax.align(
                y_val_pred_sarimax, join="inner"
            )
        logger.info("ℹ️ Обучение SARIMAX и прогнозирование завершены")


        # %%
        # Catboost (+sarimax)
        logger.info("ℹ️ Обучение Catboost и прогнозирование начаты")

        # добавляем прогноз sarimax как признак в датасет + сдвигаем его на 1 назад, чтобы соответствовать временным точкам прочих признаков
        X_train["sarima_forecast"] = sarima_forecast.shift(-1).reindex(X_train.index)
        
        # %%
        # проверим первый фолд на константность - если да отбросим его (внутри gridsearch ниже это неудобно делать)
        tss_check = TimeSeriesSplit(n_splits=n_splits)
        first_train_batch_indices, _ = next(tss_check.split(X_train, y_train))

        y_train_first_fold_segment = y_train.iloc[first_train_batch_indices]
        if len(np.unique(y_train_first_fold_segment)) == 1:
            logger.warning(f"⚠️ Фолд 0 пропускаем — целевой признак константный")
            logger.warning(f'⚠️ Даты фолда: {X_train.iloc[first_train_batch_indices].index}')
            
            # если константный, удаляем этот блок данных из X_train и y_train
            num_rows_to_skip = len(first_train_batch_indices)

            y_train = y_train.iloc[num_rows_to_skip:]
            X_train = X_train.iloc[num_rows_to_skip:]

            # после удаления первого фолда нужно будет использовать n_splits-1 для gridsearch
            n_splits -= 1

        # %%
        # Построение и обучение модели

        # подготовим обучение
        tss = TimeSeriesSplit(n_splits=n_splits)

        # итоговый пайплайн
        pipe_final = Pipeline(
            [
                (
                    "model",
                    CatBoostRegressor(
                        silent=True, 
                        random_seed=RANDOM_STATE, 
                        cat_features=cat_features,
                        thread_count=1
                                    ),
                )
            ]
        )

        # сетка гиперпараметров
        param_grid = [
            # CatBoostRegressor
            {
                "model__iterations": [100, 200, 500],
                "model__depth": [2, 5, 7],
                "model__learning_rate": [0.1, 0.001],
                "model__l2_leaf_reg": [2],
                "model__loss_function": ["RMSE"],
                "model__max_bin": [256],
                "model__random_strength": [1],
                "model__early_stopping_rounds": [10],
            }
        ]
        
        # %%
        # полный перебор гиперпараметров с помощью GridSearchCV
        grid_search = GridSearchCV(
            pipe_final,
            param_grid=param_grid,
            cv=tss,
            scoring={"neg_mean_squared_error": "neg_mean_squared_error"},
            refit="neg_mean_squared_error",
            n_jobs=1
            # error_score=np.nan
        )

        # обучение модели
        model = grid_search.fit(X_train, y_train)
        
        # обрезаем отрицательные прогнозы
        y_pred_train = np.clip(model.predict(X_train), 0, None)

        # считаем метрики
        best_score_rmse = round(np.sqrt(mean_squared_error(y_train, y_pred_train)), 4)
        best_score_mase = round(mase(y_train, y_pred_train, y_train), 4)

        # %%
        # валидация модели
        # добавляем прогнозы sarimax в валидационный датафрейм
        X_val["sarima_forecast"] = sarima_forecast.shift(-1).reindex(X_val.index)

        # переобучаем лучшую модель на всем тренировочном наборе
        final_model_cb = grid_search.best_estimator_
        final_model_cb.fit(X_train, y_train)

        # делаем прогноз и считаем метрики RMSE и MASE на валидационном наборе
        y_val_pred_cb = np.clip(np.round(final_model_cb.predict(X_val), decimals=2), 0, None)

        # считаем метрики
        rmse_val_cb = np.sqrt(mean_squared_error(y_val, y_val_pred_cb))
        mase_metrics_val_cb = mase(y_val, y_val_pred_cb, y_train)

        logger.info("ℹ️ Обучение Catboost и прогнозирование завершены")
        # %%
        # подбор модели ансамбля прогнозов на валидационной выборке
        # сверим размерности и временые ряды истинных валидационных значений и предсказаний разных моделей:

        # LSTM
        if y_val_original.shape != y_val_pred_lstm.shape:
            logger.error("❌ Размерности временных рядов LSTM не совпадают")
            raise ValueError("Размерности временных рядов LSTM не совпадают")

        # SARIMAX
        if y_val_sarimax.shape != y_val_pred_sarimax.shape:
            logger.error("❌ Размерности временных рядов SARIMAX не совпадают")
            raise ValueError("Размерности временных рядов SARIMAX не совпадают")

        # Catboost
        if y_val.shape != y_val_pred_cb.shape:
            logger.error("❌ Размерности временных рядов Catboost не совпадают")
            raise ValueError("Размерности временных рядов Catboost не совпадают")

        # сравнение между моделями
        if not (y_val_pred_lstm.shape == y_val_pred_sarimax.shape == y_val_pred_cb.shape):
            logger.error("❌ Размерности предсказаний моделей не совпадают между собой")
            raise ValueError("Размерности предсказаний моделей не совпадают между собой")

        # %%
        # собираем валидационный датафрейм
        forecasts_val = y_val.to_frame(name="y_actual")
        forecasts_val["forecast_lstm"] = pd.Series(y_val_pred_lstm, index=y_val.index)
        forecasts_val = forecasts_val.join(y_val_pred_sarimax.rename("forecast_sarimax"))
        forecasts_val["forecast_catboost"] = pd.Series(y_val_pred_cb, index=y_val.index)

        # %%
        # обучим линейную регрессию (ridge) для взвешивания прогнозов трех моделей

        # выбираем прогнозы моделей как фичи
        X_meta = forecasts_val[["forecast_lstm", "forecast_sarimax", "forecast_catboost"]]
        y_meta = forecasts_val["y_actual"]

        # обучаем Ridge-регрессию с коэффициентом регуляризации
        ridge_model = Ridge(alpha=10)
        ridge_model.fit(X_meta, y_meta)

        # %%
        # обучим простую нейросеть для взвешивания прогнозов трех моделей

        # преобразуем прогнозы на валидацинной выборке в тензоры PyTorch
        X_meta_tensor = torch.tensor(X_meta.values, dtype=torch.float32)
        y_meta_tensor = torch.tensor(y_meta.values, dtype=torch.float32)

        # задаем модель нейросети
        class SimpleNN(nn.Module):
            def __init__(self):
                super(SimpleNN, self).__init__()
                self.fc = nn.Linear(3, 1)  # 3 входных признака-прогноза

            def forward(self, x):
                return self.fc(x)

        # инициализируем модель
        model_meta = SimpleNN()

        # определяем функцию потерь и оптимизатор
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model_meta.parameters(), lr=0.001)

        # обучаем модель
        num_epochs = 500
        for epoch in range(num_epochs):
            # прямой проход
            outputs = model_meta(X_meta_tensor).squeeze()
            loss = criterion(outputs, y_meta_tensor)

            # обратный проход и оптимизация
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # %%
        # тестирование ансамбля прогнозов на тестовой выборке
        logger.info('ℹ️ Прогнозирование на тестовой выборке начато')
        
        # делаем тестовые прогнозы от модели LSTM
        if mode == "train_val":

            # добавляем последние time_steps строк из валидационного набора в тестовый
            # (тк они нужны для запуска первого прогноза, чтобы не тратить на это тестовые данные)
            X_test_full = pd.concat([X_val_prepared.tail(time_steps), X_test_prepared], axis=0)
            y_test_full = pd.concat([y_val_scaled.tail(time_steps), y_test_scaled], axis=0)

            # создаем последовательности для тестового набора
            X_seq_test, y_seq_test = create_sequences(X_test_full, y_test_full, time_steps)

            # преобразуем в тензоры
            X_test_tensor = torch.tensor(X_seq_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_seq_test, dtype=torch.float32).view(-1, 1)

            # делаем прогноз
            model_lstm.eval()
            with torch.no_grad():
                y_pred = model_lstm(X_test_tensor.to(device)).cpu().numpy()

            # разворачиваем масштабированные данные
            y_test_original = np.round(scaler_y.inverse_transform(y_test_tensor.numpy()).flatten(), decimals=2)
            y_test_pred_lstm = np.clip(np.round(scaler_y.inverse_transform(y_pred).flatten(), decimals=2),0,None)

            # преобразуем y_test_original в Series с правильными временныеми индексами
            y_test_original = pd.Series(
                y_test_original, index=y_test_full.index[-len(y_test_original) :]
            )

            # считаем метрики
            rmse_test_lstm = np.sqrt(mean_squared_error(y_test_original, y_test_pred_lstm))
            mase_test_lstm = mase(
                y_test_original, y_test_pred_lstm, pd.concat([y_train, y_val])
            )

        elif mode == "production":
            # берём последние time_steps строк, чтобы сделать прогноз на следующий день
            X_test_full = np.array(pd.concat([X_val_prepared.tail(time_steps), X_test_prepared], axis=0))

            # преобразуем в тензор и добавляем размерность batch
            X_test_tensor = torch.tensor(X_test_full, dtype=torch.float32).unsqueeze(0)

            # делаем прогноз
            model_lstm.eval()
            with torch.no_grad():
                y_pred = model_lstm(X_test_tensor.to(device)).cpu().numpy()

            # обратно из масштаба
            y_test_pred_lstm = np.clip(np.round(scaler_y.inverse_transform(y_pred).flatten()[0], decimals=2),0,None)

        # %%
        # отбираем тестовые прогнозы от модели SARIMAX

        if mode == "train_val":

            # делим данные
            y_test_sarimax = data_fund.shift(-1).loc[data_fund.index >= test_start_date, "day_payments_sum"].copy().asfreq("D")
            y_test_pred_sarimax = sarima_forecast.shift(-1).loc[sarima_forecast.index >= test_start_date].copy().asfreq("D")
            
            y_test_sarimax = y_test_sarimax.dropna()
            y_test_pred_sarimax = y_test_pred_sarimax.dropna()

            # выравниванием и отбрасываем NaN
            y_test_sarimax, y_test_pred_sarimax = y_test_sarimax.align(y_test_pred_sarimax, join="inner")

            # считаем метрики RMSE и MASE
            rmse_test_sarimax = np.sqrt(mean_squared_error(y_test_sarimax, y_test_pred_sarimax))
            mase_test_sarimax = mase(y_test_sarimax, y_test_pred_sarimax, pd.concat([y_train, y_val]))

        elif mode == "production":

            # делим данные
            forecast = sarimax_results.get_forecast(steps=1)
            y_test_pred_sarimax = max(forecast.predicted_mean.item(), 0)
            
        # %%
        # делаем тестовые прогнозы от модели Catboost

        # добавляем прогнозы sarimax в тестовый датафрейм, сдвинув ихз на -1 для соответствия с признаками catboost
        X_test["sarima_forecast"] = sarima_forecast.shift(-1).reindex(X_test.index)

        # делаем прогноз и считаем метрики RMSE и MASE на тестовом наборе
        y_test_pred_cb = np.clip(np.round(final_model_cb.predict(X_test), decimals=2), 0, None)

        if mode == "train_val":
            rmse_test_cb = np.sqrt(mean_squared_error(y_test, y_test_pred_cb))
            mase_metrics_test_cb = mase(y_test, y_test_pred_cb, pd.concat([y_train, y_val]))
        elif mode == "production":
            y_test_pred_cb = float(y_test_pred_cb[0])

        # %%
        # собираем тестовые прогнозы в датафрейм
        if mode == "train_val":
            forecasts_test = pd.DataFrame(
                {
                    "fund_id": FUND_ID,
                    "y_actual": y_test,
                    "forecast_lstm": pd.Series(y_test_pred_lstm, index=y_test.index),
                    "forecast_sarimax": pd.Series(y_test_pred_sarimax, index=y_test.index),
                    "forecast_catboost": pd.Series(y_test_pred_cb, index=y_test.index),
                },
                index=y_test.index
            )

        elif mode == "production":
            predict_date = data_fund.index[-1] + pd.Timedelta(days=1)
            forecasts_test = pd.DataFrame(
                {
                    "fund_id": FUND_ID,
                    "forecast_lstm": pd.Series(y_test_pred_lstm, index=[predict_date]),
                    "forecast_sarimax": pd.Series(y_test_pred_sarimax, index=[predict_date]),
                    "forecast_catboost": pd.Series(y_test_pred_cb, index=[predict_date]),
                },
                index=[predict_date]
            )

        # %%
        # предсказание ансамблем с взвешиванием линейной регрессией

        if mode == "train_val":
            # выбираем прогнозы моделей как фичи
            X_meta_test = forecasts_test[
                ["forecast_lstm", "forecast_sarimax", "forecast_catboost"]
            ]
            
            forecasts_test["forecast_ridge"] = np.round(np.clip(ridge_model.predict(X_meta_test), 0, None),2)

            ens_linear_rmse = np.sqrt(
                mean_squared_error(
                    forecasts_test["y_actual"], forecasts_test["forecast_ridge"]
                )
            )
            ens_linear_mase = mase(
                forecasts_test["y_actual"],
                forecasts_test["forecast_ridge"],
                pd.concat([y_train, y_val]),
            )

        elif mode == "production":
            X_meta_test = forecasts_test[
                ["forecast_lstm", "forecast_sarimax", "forecast_catboost"]
            ]
            
            forecasts_test["forecast_ridge"] = np.round(np.clip(ridge_model.predict(X_meta_test).item(), 0, None),2)

        # %%
        # предсказание ансамблем с взвешиванием нейронкой

        if mode == "train_val":
            # преобразуем данные в тензоры PyTorch
            X_meta_test_tensor = torch.tensor(X_meta_test.values, dtype=torch.float32)

            # прогнозируем на тестовых данных
            forecasts_test["forecast_nn"] = np.round(np.clip(model_meta(X_meta_test_tensor).detach().numpy(), 0, None),2)

            ens_nn_rmse = np.sqrt(mean_squared_error(forecasts_test["y_actual"], forecasts_test["forecast_nn"]))
            ens_nn_mase = mase(forecasts_test["y_actual"],forecasts_test["forecast_nn"],pd.concat([y_train, y_val]))

        elif mode == "production":

            # преобразуем данные в тензоры PyTorch
            X_meta_test_tensor = torch.tensor(X_meta_test.values, dtype=torch.float32)

            # прогнозируем на тестовых данных
            forecasts_test["forecast_nn"] = np.round(np.clip(model_meta(X_meta_test_tensor).detach().numpy().item(), 0, None),2)
        
        logger.info('ℹ️ Прогнозирование на тестовой выборке завершено')

        # %%
        if mode == "train_val":
            path3 = "comparison_train_val.csv"
            if os.path.exists(path3):
                comparison = pd.read_csv(path3)
            else:
                comparison = pd.DataFrame(columns=["fund_id"])

            # проверяем, есть ли уже этот фонд
            if FUND_ID in comparison["fund_id"].values:
                # обновляем существующую строку
                comparison.loc[comparison["fund_id"] == FUND_ID, "3models_lstm_rmse"] = (rmse_test_lstm)
                comparison.loc[comparison["fund_id"] == FUND_ID, "3models_lstm_mase"] = (mase_test_lstm)
                comparison.loc[comparison["fund_id"] == FUND_ID, "3models_sarimax_rmse"] = (rmse_test_sarimax)
                comparison.loc[comparison["fund_id"] == FUND_ID, "3models_sarimax_mase"] = (mase_test_sarimax)
                comparison.loc[comparison["fund_id"] == FUND_ID, "3models_catboost_rmse"] = rmse_test_cb
                comparison.loc[comparison["fund_id"] == FUND_ID, "3models_catboost_mase"] = mase_metrics_test_cb
                comparison.loc[comparison["fund_id"] == FUND_ID, "3models_ens_linear_rmse"] = ens_linear_rmse
                comparison.loc[comparison["fund_id"] == FUND_ID, "3models_ens_linear_mase"] = ens_linear_mase
                comparison.loc[comparison["fund_id"] == FUND_ID, "3models_ens_nn_rmse"] = (ens_nn_rmse)
                comparison.loc[comparison["fund_id"] == FUND_ID, "3models_ens_nn_mase"] = (ens_nn_mase)
            else:
                # создаем новую строку для 3 моделей
                new_row = pd.DataFrame(
                    {
                        "fund_id": [FUND_ID],
                        "3models_lstm_rmse": [rmse_test_lstm],
                        "3models_lstm_mase": [mase_test_lstm],
                        "3models_sarimax_rmse": [rmse_test_sarimax],
                        "3models_sarimax_mase": [mase_test_sarimax],
                        "3models_catboost_rmse": [rmse_test_cb],
                        "3models_catboost_mase": [mase_metrics_test_cb],
                        "3models_ens_linear_rmse": [ens_linear_rmse],
                        "3models_ens_linear_mase": [ens_linear_mase],
                        "3models_ens_nn_rmse": [ens_nn_rmse],
                        "3models_ens_nn_mase": [ens_nn_mase],
                    }
                )

                comparison = pd.concat([comparison, new_row], ignore_index=True)

            # сохраняем обратно
            comparison.to_csv(path3, index=False)
            logger.info(f'ℹ️ Данные метрик в режиме train_val сохранены в файл {path3}')

        elif mode == "production":

            path4 = "forecasts_full.csv"
            file_exists = os.path.isfile(path4)

            # сохраняем с добавлением (append), пишем заголовки только если файл создаётся впервые
            forecasts_test.to_csv(path4, mode="a", header=not file_exists, index=True)
            logger.info(f'ℹ️ Данные прогнозов моделей в режиме production сохранены в файл {path4}')

    # формируем итоговый прогноз из данных модели и медианных значений для фондов с недостаточной информацией
    if mode == 'production':
        # читаем предсказания нейросети
        forecast_date = predict_date.strftime('%Y-%m-%d')

        forecasts_df = pd.read_csv(path4, index_col=0)
        forecast_tomorrow = (forecasts_df[forecasts_df.index == forecast_date][['fund_id', 'forecast_nn']]
                            .rename(columns={'forecast_nn': 'forecast'}))

        # отбираем фонды для медианного прогноза
        other_funds = list(set(data_final['user_id'].unique()) - set(forecast_tomorrow['fund_id'].unique()))
        median_rows = []

        # считаем медианы
        for o_fund in other_funds:
            median_val = data_final[data_final['user_id'] == o_fund]['day_payments_sum'].median()
            median_rows.append({
                'fund_id': o_fund,
                'forecast': round(median_val, 2)
            })
        # собираем медианы в датафрейм
        median_forecast_df = pd.DataFrame(median_rows)
        median_forecast_df.index = [forecast_date] * len(median_forecast_df)

        # соединяем прогнозы
        forecast_full = pd.concat([forecast_tomorrow, median_forecast_df], axis=0)
        forecast_full = forecast_full.reset_index().sort_values(by=['index', 'fund_id']).set_index('index')
        forecast_full.index.name = None

        path5 = 'forecast_tomorrow.csv'
        if os.path.exists(path5):
            df_old = pd.read_csv(path5, index_col=0)
            df = pd.concat([df_old, forecast_full], axis=0)
        else:
            df = forecast_full

        df.to_csv(path5)
        logger.info(f'ℹ️ Данные прогнозов моделей и средних значений в режиме production сохранены в файл {path5}')

    # проверка и удаление файлов с эмбеддингами (пока — потом будем их наращивать)
    #for fname in ['2try_data_with_purpose_mean.parquet', '2try_data_with_purpose_lemma.parquet', '2try_data_with_purpose_emb.parquet']:
    #    if os.path.exists(fname):
    #        os.remove(fname)
    logger.info(f"✅ Завершение скрипта")
