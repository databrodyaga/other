# %%
#!/usr/bin/env python3

# Прогнозирование платежей с применением ансамбля моделей на реальных данных (МЕСЯЧНАЯ АГРЕГАЦИЯ)

# Прогнозируем суммы платежей на агрегированных данных на следующий месяц c учетом временной составляющей с применением трех моделей: LSTM, SARIMAX, Catboost и последующим взвешиванием прогнозов с помощью нейронной сети.

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

    # Определяем конец периода как начало текущего месяца минус один день (последний полный месяц)
    # Или просто первый день текущего месяца, если данные обновляются live. 
    # Предполагаем, что хотим закрыть по последний полный месяц.
    end_date = pd.Timestamp.today().replace(day=1) - pd.DateOffset(days=1)
    end_date = end_date.replace(day=1) # Снова к первому числу
    
    # Если в данных есть даты позже расчетной (например, текущий неполный месяц), берем максимум из данных
    if group['date'].max() > end_date:
        end_date = group['date'].max()

    # Создаем диапазон с частотой 'MS' (Month Start)
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
# адаптировано: проверяем наличие платежей, но пороги другие для месяцев
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

# %%

logger.info(f"✅ Запуск скрипта (Месячная агрегация)")

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
ROLLING_WINDOW = 3 # размер окна скользящего среднего (квартал)
LAGS = [1, 2, 3, 6, 12] # количество лагов (месяц, квартал, полгода, год)

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
        #"date_to": "2025-12-10"   
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
# конвертируем дату в datetime
data["date"] = pd.to_datetime(data["date"], yearfirst=True, errors='coerce')

# и убираем записи из будущего и далекого прошлого, меньше нуля (и такое бывает) и без привязки к фондам (user_id == 0)
yesterday = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
data = data[data["date"] <= yesterday]
data = data[data["payments_amount"] > 0]
data = data[data["date"] >= "2015-01-01"]
data = data[data["user_id"] != 0]

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
# агрегация платежей по МЕСЯЦАМ и пользователям
logger.info('ℹ️ Агрегируем атомарные данные до уровня месяцев')

# Приводим дату к началу месяца (Month Start) для агрегации
data_fe["month_date"] = data_fe["date"].astype('datetime64[M]')

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

# агрегируем данные до уровня МЕСЯЦА
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
    .rename(columns={"month_date": "date"}) # Возвращаем имя 'date' для совместимости кода
)

# заполним NaN в std колонках нулями (там где было одно наблюдение/платеж)
std_cols = [col for col in data_monthly_agg.columns if col.endswith("_std")]
data_monthly_agg[std_cols] = data_monthly_agg[std_cols].fillna(0)
# и округлим
for col in data_monthly_agg.columns:
    if col.endswith("_ufreq_mean") or col.endswith("_ufreq_std"):
        data_monthly_agg[col] = data_monthly_agg[col].round(6)
    elif col in ["month_payments_sum", "month_min_payment", "month_max_payment"]:
        data_monthly_agg[col] = data_monthly_agg[col].round(2)

# %%
# заполним пропущенные МЕСЯЦЫ нулевыми значениями, чтобы сохранить структуру временного ряда
logger.info('ℹ️ Заполняем пропуски в агрегированных данных (месячный временной ряд) нулями')
data_monthly_agg_filled = data_monthly_agg.groupby("user_id", group_keys=False).apply(fill_missing_dates)

# добавляем столбцы с номером месяца и кварталом + кодируем их с помощью циклического кодирования (sin/cos)
logger.info('ℹ️ Добавляем временные признаки (месяц, квартал)')

# Месяц года (1-12)
data_monthly_agg_filled["month_of_year"] = data_monthly_agg_filled["date"].dt.month
data_monthly_agg_filled["month_sin"] = np.sin(2 * np.pi * data_monthly_agg_filled["month_of_year"] / 12).round(6)
data_monthly_agg_filled["month_cos"] = np.cos(2 * np.pi * data_monthly_agg_filled["month_of_year"] / 12).round(6)

# Квартал (1-4)
data_monthly_agg_filled["quarter"] = data_monthly_agg_filled["date"].dt.quarter

# %%
# добавим по платежам скользящее среднее и лаги
logger.info('ℹ️ Добавляем скользящие средние и лаги по целевому признаку')

# применим функцию скользящего среднего к каждой группе после группировки по user_id
data_monthly_agg_filled_added = data_monthly_agg_filled.groupby(
    "user_id", group_keys=False
).apply(lambda group: rolling_mean(group, rolling_window=ROLLING_WINDOW, lags=LAGS))

data_monthly_agg_filled_added = data_monthly_agg_filled_added.fillna(0)

# %%
# преобразуем дату в индекс
data_final = data_monthly_agg_filled_added.set_index("date", drop=True)
data_final.index.name = None

# %%
# удаляем лишние колонки (префиксы теперь month_)
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

## сохраним/обновим фактические данные для дальнейшего анализа и расчета метрик (теперь месячные)
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
data_final.to_csv('data_monthly_payments.csv')

# %%
# запускаем обучение моделей
# отберем для обучения фонды, для которых количество МЕСЯЦЕВ с данными больше заданного порога 

data_final = data_final.sort_index()

# подсчёт количества активных месяцев
months_per_user_id = data_final[data_final["month_payments_sum"] != 0].groupby("user_id")['month_payments_sum'].count()

# проверка последних 3 месяцев на наличие данных (чтобы не прогнозировать мертвые фонды)
months_per_user_nonzero = (
    data_final
    .sort_index()
    .groupby('user_id')
    .apply(lambda x: (x.iloc[-3:]['month_payments_sum'] != 0).sum())
)

# фильрация по количеству активных месяцев (ставим 18 месяцев - 1.5 года для сезонности)
funds = months_per_user_id[
    (months_per_user_id >= 18) &
    (months_per_user_nonzero > 0)
].index.tolist()

logger.info(f'ℹ️ Фильтрация по количеству активных месяцев и наличию платежей проведена, отобраны следующие фонды: {funds}')
#funds = [176,784,791,799,886,1001] #237

# %%
# запускаем общий цикл обучения и прогнозирования по списку отобранных фондов
logger.info('ℹ️ Запущен общий цикл обучения и прогнозирования по списку отобранных фондов')

mode = "train_val"  # production/train_val
logger.info(f"ℹ️ Режим прогнозирования: {mode}")

for fund in tqdm(funds):

    # %%
    # выберем фонд
    FUND_ID = fund
    logger.info(f"🔹 FUND_ID: {FUND_ID}")

    # устанавливаем сиды - для каждого фонда свой(для torch)
    user_specific_seed = RANDOM_STATE + int(FUND_ID) 
    seed_all(user_specific_seed)

    data_fund = data_final[data_final["user_id"] == FUND_ID].copy()

    # проверяем монотонность и сортируем при необходимости
    if not data_fund.index.is_monotonic_increasing:
        data_fund = data_fund.sort_index()

    # проверяем ряд на стартовые нули и разреженные данные
    data_fund = start_date_define(data_fund, target_col="month_payments_sum")

    if data_fund is None:
        logger.warning(f"⚠️ Фонд {FUND_ID} пропускаем — ряд данных слишком разреженный")
        continue

    # убираем user_id
    data_fund.drop(["user_id"], axis=1, inplace=True)

    # %%
    # заменяем агрегированные месячные суммы из THRESHOLD_CONST перцентиля на медианное значение
    # (для сглаживания выбросов)
    TRESHOLD_CONST = 0.95 # чуть мягче для месяца, т.к. суммы могут сильно варьироваться

    threshold = data_fund["month_payments_sum"].quantile(TRESHOLD_CONST)
    data_fund.loc[data_fund["month_payments_sum"] > threshold, "month_payments_sum"] = data_fund["month_payments_sum"].median()

    # Оставляем только полные месяцы (все, что строго меньше начала текущего месяца)
    current_month_start = pd.Timestamp.today().replace(day=1)
    
    # Если в данных есть даты больше или равные текущему месяцу — отрезаем их
    data_fund = data_fund[data_fund.index < current_month_start]

    # %%

    # создаём версию со сдвигом для train/val
    data_fund_shifted = data_fund.copy()
    data_fund_shifted["target_shifted"] = data_fund_shifted["month_payments_sum"].shift(
        -1
    )
    data_fund_shifted = data_fund_shifted[:-1]  # убираем последнюю строку с NaN

    # делим данные на тренировочные, валидационные и тестовые (в МЕСЯЦАХ)
    if mode == "train_val":
        #test_start_date = data_fund_shifted.index.max() # Последний месяц для теста
        test_start_date = data_fund_shifted.index.sort_values()[-2] # 2. есяца а тест
        val_start_date = (test_start_date - pd.DateOffset(months=3)) # 3 месяцев на валидацию
        val_end_date = test_start_date - pd.DateOffset(months=1)

        # делим данные
        train_data_fund = data_fund_shifted.loc[: val_start_date - pd.DateOffset(months=1)]
        val_data_fund = data_fund_shifted.loc[val_start_date:val_end_date]
        test_data_fund = data_fund_shifted.loc[test_start_date:]

    elif mode == "production":
        test_start_date = data_fund.index.max() # Текущий последний месяц
        # Для продакшена валидацию берем за 2 мес до последнего месяца
        val_start_date = test_start_date - pd.DateOffset(months=3)

        # делим данные
        train_data_fund = data_fund_shifted.loc[: val_start_date - pd.DateOffset(months=1)]
        val_data_fund = data_fund_shifted.loc[
            val_start_date : test_start_date - pd.DateOffset(months=1)
        ]
        test_data_fund = data_fund.loc[
            [test_start_date]
        ]  # одна строка (последний месяц) для прогноза на следующий

    # проверяем фиксируем размеры и даты
    logger.debug(f'🔍 Размеры для LSTM/Catboost: train={train_data_fund.shape}, val={val_data_fund.shape}, test={test_data_fund.shape}')
    
    # %%
    # выделяем категоральные признаки (для CatBoost)
    cat_features = ["month_of_year"] # День недели больше не актуален

    # формируем X_train и y_train
    X_train = train_data_fund.drop(["month_payments_sum", "target_shifted"], axis=1)
    y_train = train_data_fund["target_shifted"]

    # формируем X_val и y_val
    X_val = val_data_fund.drop(["month_payments_sum", "target_shifted"], axis=1)
    y_val = val_data_fund["target_shifted"]

    # формируем X_test и y_test
    if mode == "train_val":
        X_test = test_data_fund.drop(["month_payments_sum", "target_shifted"], axis=1)
        y_test = test_data_fund["target_shifted"]
    elif mode == "production":
        X_test = test_data_fund.drop(["month_payments_sum"], axis=1)

    # %%
    # LSTM
    # Подготовка данных

    # масштабируем числовые признаки (убираем cat_features из скейлинга, если они не закодированы)
    scaler_x = StandardScaler()
    
    # Разделяем признаки
    X_train_num = X_train.drop(columns=cat_features)
    X_val_num = X_val.drop(columns=cat_features)
    X_test_num = X_test.drop(columns=cat_features)
    
    X_train_num_scaled = scaler_x.fit_transform(X_train_num)
    X_val_num_scaled = scaler_x.transform(X_val_num)
    X_test_num_scaled = scaler_x.transform(X_test_num)

    # кодируем категориальные признаки (One-Hot для LSTM)
    X_train_cat_encoded = pd.get_dummies(X_train[cat_features].astype(str), dtype=float)
    X_val_cat_encoded = pd.get_dummies(X_val[cat_features].astype(str), dtype=float)
    X_test_cat_encoded = pd.get_dummies(X_test[cat_features].astype(str), dtype=float)

    # Выравниваем столбцы
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
    cols_prepared = list(X_train_num.columns) + list(X_train_cat_encoded.columns)
    
    X_train_prepared = pd.DataFrame(X_train_prepared, columns=cols_prepared, index=X_train.index)
    X_val_prepared = pd.DataFrame(X_val_prepared, columns=cols_prepared, index=X_val.index)
    X_test_prepared = pd.DataFrame(X_test_prepared, columns=cols_prepared, index=X_test.index)

    # преобразуем в Series
    y_train_scaled = pd.Series(y_train_scaled.ravel(), name=y_train.name, index=y_train.index)
    y_val_scaled = pd.Series(y_val_scaled.ravel(), name=y_val.name, index=y_val.index)
    if mode == "train_val":
        y_test_scaled = pd.Series(y_test_scaled.ravel(), name=y_test.name, index=y_test.index)

    # %%
    # Построение и обучение модели LSTM
    logger.info("ℹ️ Обучение LSTM и прогнозирование начаты")
    # %%
    # задаем шаги (увеличиваем для месячных, чтобы охватить год)
    time_steps = 12 
    n_splits = 3 # уменьшаем кол-во сплитов, т.к. данных меньше

    # задаем устройство для обучения
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    # Проверка, хватает ли данных для sequence
    if len(X_seq_full) == 0:
        logger.warning(f"⚠️ Недостаточно данных для LSTM sequence (time_steps={time_steps}). Фонд пропускаем.")
        continue

    # преобразуем в тензоры
    X_full_tensor = torch.tensor(X_seq_full, dtype=torch.float32).to(device)
    y_full_tensor = torch.tensor(y_seq_full, dtype=torch.float32).view(-1, 1).to(device)

    # фомируем модель LSTM
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(LSTMModel, self).__init__()
            self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
            #self.dropout1 = nn.Dropout(0.2) # Добавим dropout т.к. данных мало
            self.fc1 = nn.Linear(hidden_size, 128)
            self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
            self.fc2 = nn.Linear(128, output_size)

        def forward(self, x):
            x, _ = self.lstm1(x)
            #x = self.dropout1(x)
            x = x[:, -1, :]
            x = self.fc1(x)
            x = self.leaky_relu(x)
            x = self.fc2(x)
            x = self.leaky_relu(x)
            return x

    # инициализация модели (уменьшим скрытый слой)
    input_size = X_seq_full.shape[2]
    hidden_size = 128 
    output_size = 1
    model_lstm = LSTMModel(input_size, hidden_size, output_size).to(device)

    # задаем функцию потерь и оптимизатор
    criterion = nn.HuberLoss(delta=1.5)
    optimizer = optim.AdamW(model_lstm.parameters(), lr=0.002, weight_decay=1e-4)

    # обучаем модель
    num_epochs = 300 # Увеличим эпохи, т.к. батчей мало
    for epoch in range(num_epochs):
        model_lstm.train()
        optimizer.zero_grad()
        output = model_lstm(X_full_tensor)
        loss = criterion(output, y_full_tensor)
        loss.backward()
        optimizer.step()

    # %%
    # добавляем последние time_steps строк из тренировочного набора в валидационный набор
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
    # подготовка данных. загружаем ряд с несдвинутым целевым признаком

    if mode == "train_val":
        #test_start_date = data_fund_shifted.index.max()
        #val_start_date = (test_start_date - pd.DateOffset(months=6))
        #val_end_date = test_start_date - pd.DateOffset(months=1)
        
        # последний месяц в данных
        test_end_date = data_fund_shifted.index.max()
        test_start_date = test_end_date - pd.DateOffset(months=1)
        val_end_date = test_start_date - pd.DateOffset(months=1)
        val_start_date = val_end_date - pd.DateOffset(months=2)

        y_train_sarimax = data_fund.loc[data_fund.index < val_start_date, "month_payments_sum"].copy()
        y_val_sarimax = data_fund.loc[val_start_date:val_end_date, "month_payments_sum"].copy()
        y_test_sarimax = data_fund.loc[data_fund.index >= test_start_date, "month_payments_sum"].copy()

    elif mode == "production":
        test_start_date = data_fund.index.max()
        val_start_date = test_start_date - pd.DateOffset(months=2)

        y_train_sarimax = data_fund.loc[data_fund.index < val_start_date, "month_payments_sum"].copy()
        y_val_sarimax = data_fund.loc[val_start_date : test_start_date - pd.DateOffset(months=1), "month_payments_sum"].copy()
        y_test_sarimax = data_fund.loc[[test_start_date], "month_payments_sum"].copy()

    logger.debug(f'🔍 Размеры SARIMAX: train={y_train_sarimax.shape}, val={y_val_sarimax.shape}, test={y_test_sarimax.shape}')

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

        data = data.asfreq("MS") # Month Start frequency

        for fold, (train_index, test_index) in enumerate(tscv.split(data)):
            train = data.iloc[train_index]
            test = data.iloc[test_index]
            
            # Пропускаем, если слишком мало данных для сезонности
            if len(train) < 13:
                 continue

            if len(np.unique(train)) == 1:
                continue

            try:
                model_sarimax = SARIMAX(
                    train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False, # Ослабляем требования для малых данных
                    enforce_invertibility=False,
                )

                results = model_sarimax.fit(
                    disp=False,
                    maxiter=500,
                    optim_score="harvey",
                    method="powell"
                )

                forecast = results.get_forecast(steps=len(test))
                predicted_mean = forecast.predicted_mean.clip(lower=0).round(2)

                mask = ~predicted_mean.isna() & ~test.isna()
                predicted_clean = predicted_mean[mask]
                test_clean = test[mask]

                if len(test_clean) > 0:
                    current_rmse = np.sqrt(mean_squared_error(test_clean, predicted_clean))
                    rmse_scores.append(current_rmse)
                    
                    current_mase = mase(test_clean, predicted_clean, train, seasonality=12)
                    mase_scores.append(current_mase)

                    if current_rmse < best_rmse:
                        best_rmse = current_rmse
                        best_model = results
            except Exception as e:
                # logger.debug(f"⚠️ Ошибка SARIMAX fold {fold}: {e}")
                continue

        return np.nanmean(rmse_scores), np.nanmean(mase_scores), best_model

    # задаем наборы гиперпараметров для SARIMAX (Сезонность 12!)
    param_sets = [
        ((1, 1, 0), (0, 1, 1, 12)), 
        ((0, 1, 1), (0, 1, 1, 12)),
        ((1, 0, 0), (1, 1, 0, 12)),
        ((0, 1, 1), (1, 1, 1, 12)),
        ((1, 1, 0), (0, 0, 0, 0)), # Без сезонности, если данные короткие
    ]

    best_model_overall_sarimax = None
    best_rmse_overall_sarimax = np.inf
    
    # Дефолтная модель, если ничего не подберется
    default_model = SARIMAX(y_train_sarimax.asfreq("MS"), order=(1,1,0), enforce_stationarity=False).fit(disp=False)
    best_model_overall_sarimax = default_model

    for param in param_sets:
        average_rmse_sarimax, average_mase_sarimax, best_model_sarimax = (
            cross_val_sarima(y_train_sarimax, param[0], param[1])
        )

        if average_rmse_sarimax < best_rmse_overall_sarimax and best_model_sarimax is not None:
            best_rmse_overall_sarimax = average_rmse_sarimax
            best_model_overall_sarimax = best_model_sarimax

    # %%
    # переобучаем лучшую модель на всем датасете инкрементально
    y_full_sarimax = data_fund["month_payments_sum"].copy().asfreq("MS")
    sarima_forecast = pd.Series(index=y_full_sarimax.index, dtype=float)

    # Минимальный размер для старта (3 месяца)
    min_full_size = 3 
    refit_interval = 1 # Переобучаем каждый месяц, т.к. точек мало

    # Заполняем начало средним
    sarima_forecast.iloc[0] = y_full_sarimax.iloc[0]
    for i in range(1, min(min_full_size, len(y_full_sarimax))):
        sarima_forecast.iloc[i] = round(y_full_sarimax.iloc[:i].mean(), 2)

    order = best_model_overall_sarimax.model.order
    seasonal_order = best_model_overall_sarimax.model.seasonal_order

    # Инкрементальное обучение
    if len(y_full_sarimax) > min_full_size:
        for i in range(min_full_size, len(y_full_sarimax)):
            
            # Полное переобучение всегда (данных мало, apply может быть нестабилен)
            train_series = y_full_sarimax.iloc[:i]
            try:
                model_sarimax = SARIMAX(
                    train_series,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                sarimax_results = model_sarimax.fit(disp=False, maxiter=200, method="powell")
                
                pred = sarimax_results.get_forecast(steps=1).predicted_mean.item()
                sarima_forecast.iloc[i] = max(0, round(pred, 2))
                
                # Сохраняем последнюю модель для production прогноза
                if i == len(y_full_sarimax) - 1:
                    last_sarimax_results = sarimax_results
            except:
                sarima_forecast.iloc[i] = sarima_forecast.iloc[i-1] # fallback

    # обнуляем начало
    sarima_forecast.iloc[:min_full_size] = np.nan

    # %%
    # валидация модели
    if mode == "train_val":
        y_val_sarimax = (
            data_fund.shift(-1)
            .loc[val_start_date:val_end_date, "month_payments_sum"]
            .copy()
            .asfreq("MS")
        )
        y_val_pred_sarimax = (
            sarima_forecast.shift(-1)
            .loc[val_start_date:val_end_date]
            .copy()
            .asfreq("MS")
        )
        y_val_sarimax, y_val_pred_sarimax = y_val_sarimax.align(
            y_val_pred_sarimax, join="inner"
        )

    elif mode == "production":
        y_val_sarimax = (
            data_fund.shift(-1)
            .loc[
                val_start_date : test_start_date - pd.DateOffset(months=1),
                "month_payments_sum",
            ]
            .copy()
            .asfreq("MS")
        )
        y_val_pred_sarimax = (
            sarima_forecast.shift(-1)
            .loc[val_start_date : test_start_date - pd.DateOffset(months=1)]
            .copy()
            .asfreq("MS")
        )
        y_val_sarimax, y_val_pred_sarimax = y_val_sarimax.align(
            y_val_pred_sarimax, join="inner"
        )
    logger.info("ℹ️ Обучение SARIMAX и прогнозирование завершены")


    # %%
    # Catboost
    logger.info("ℹ️ Обучение Catboost и прогнозирование начаты")

    # добавляем прогноз sarimax как признак
    X_train["sarima_forecast"] = sarima_forecast.shift(-1).reindex(X_train.index)

    # %%
    # Проверка фолдов
    if len(X_train) > n_splits + 1:
        tss = TimeSeriesSplit(n_splits=n_splits)
    else:
        # Если данных очень мало, используем hold-out без кросс-валидации
        tss = None

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

    param_grid = [
        {
            "model__iterations": [100, 300],
            "model__depth": [2, 4],
            "model__learning_rate": [0.05, 0.01],
            "model__l2_leaf_reg": [3],
            "model__loss_function": ["RMSE"]
        }
    ]

    # %%
    # GridSearch
    if tss:
        grid_search = GridSearchCV(
            pipe_final,
            param_grid=param_grid,
            cv=tss,
            scoring={"neg_mean_squared_error": "neg_mean_squared_error"},
            refit="neg_mean_squared_error",
            n_jobs=1
        )
        model = grid_search.fit(X_train, y_train)
        final_model_cb = grid_search.best_estimator_
    else:
        # Просто обучаем без подбора, если мало данных
        final_model_cb = pipe_final.fit(X_train, y_train)

    # валидация модели
    X_val["sarima_forecast"] = sarima_forecast.shift(-1).reindex(X_val.index)
    
    # Заполняем пропуски в саримаксе (начальный период) нулями или средним, чтобы не падал катбуст
    X_val["sarima_forecast"] = X_val["sarima_forecast"].fillna(0)
    X_train["sarima_forecast"] = X_train["sarima_forecast"].fillna(0)

    # Переобучаем на всем трейне
    final_model_cb.fit(X_train, y_train)

    # Прогноз
    y_val_pred_cb = np.clip(np.round(final_model_cb.predict(X_val), decimals=2), 0, None)

    logger.info("ℹ️ Обучение Catboost и прогнозирование завершены")
    
    # %%
    # Ансамбль

    # Выравнивание длин (SARIMAX мог отрезать больше из-за лагов/окон)
    common_idx = y_val_original.index.intersection(y_val_sarimax.index).intersection(y_val.index)
    
    y_val_ens = y_val.loc[common_idx]
    y_val_pred_lstm_ens = pd.Series(y_val_pred_lstm, index=y_val.index).loc[common_idx]
    y_val_pred_sarimax_ens = y_val_pred_sarimax.loc[common_idx]
    y_val_pred_cb_ens = pd.Series(y_val_pred_cb, index=y_val.index).loc[common_idx]

    # собираем валидационный датафрейм
    forecasts_val = y_val_ens.to_frame(name="y_actual")
    forecasts_val["forecast_lstm"] = y_val_pred_lstm_ens
    forecasts_val["forecast_sarimax"] = y_val_pred_sarimax_ens
    forecasts_val["forecast_catboost"] = y_val_pred_cb_ens
    
    # Заполнение пропусков (если где-то NaN вылез)
    forecasts_val = forecasts_val.fillna(0)

    # %%
    # Ridge
    X_meta = forecasts_val[["forecast_lstm", "forecast_sarimax", "forecast_catboost"]]
    y_meta = forecasts_val["y_actual"]
    
    ridge_model = Ridge(alpha=10)
    ridge_model.fit(X_meta, y_meta)

    # %%
    # Neural Net Ensemble
    X_meta_tensor = torch.tensor(X_meta.values, dtype=torch.float32)
    y_meta_tensor = torch.tensor(y_meta.values, dtype=torch.float32)

    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc = nn.Linear(3, 1)

        def forward(self, x):
            return self.fc(x)

    model_meta = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_meta.parameters(), lr=0.01)

    for epoch in range(500):
        outputs = model_meta(X_meta_tensor).squeeze()
        loss = criterion(outputs, y_meta_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # %%
    # ТЕСТ / ПРОГНОЗ
    logger.info('ℹ️ Прогнозирование на тестовой выборке (будущий месяц) начато')

    # LSTM Test
    if mode == "train_val":
        # --- ИСПРАВЛЕНИЕ: Берем контекст из Train + Val, так как Val (2 мес) короче time_steps (12 мес) ---
        
        # Объединяем Train и Val, берем последние time_steps
        context_X = pd.concat([X_train_prepared, X_val_prepared], axis=0).tail(time_steps)
        context_y = pd.concat([y_train_scaled, y_val_scaled], axis=0).tail(time_steps)
        
        # Добавляем Test
        X_test_full = pd.concat([context_X, X_test_prepared], axis=0)
        y_test_full = pd.concat([context_y, y_test_scaled], axis=0)
        
        # Создаем последовательности
        X_seq_test, y_seq_test = create_sequences(X_test_full, y_test_full, time_steps)

        # Проверка на случай, если данных все равно не хватило
        if len(X_seq_test) == 0:
            logger.warning(f"⚠️ LSTM: Недостаточно истории для теста фонда {FUND_ID}. Заполняем нулями.")
            y_test_pred_lstm = np.zeros(len(y_test))
        else:
            X_test_tensor = torch.tensor(X_seq_test, dtype=torch.float32)
            model_lstm.eval()
            with torch.no_grad():
                y_pred = model_lstm(X_test_tensor.to(device)).cpu().numpy()
            y_test_pred_lstm = np.clip(np.round(scaler_y.inverse_transform(y_pred).flatten(), 2), 0, None)
        
        # Индекс
        y_test_idx = y_test.index

        # Считаем метрики
        rmse_test_lstm = np.sqrt(mean_squared_error(y_test, y_test_pred_lstm))
        mase_test_lstm = mase(y_test, y_test_pred_lstm, pd.concat([y_train, y_val]), 12)

    elif mode == "production":
        X_test_full = np.array(pd.concat([X_val_prepared.tail(time_steps), X_test_prepared], axis=0))
        X_test_tensor = torch.tensor(X_test_full, dtype=torch.float32).unsqueeze(0)
        model_lstm.eval()
        with torch.no_grad():
            y_pred = model_lstm(X_test_tensor.to(device)).cpu().numpy()
        y_test_pred_lstm = np.clip(np.round(scaler_y.inverse_transform(y_pred).flatten()[0], decimals=2),0,None)
        y_test_idx = [data_fund.index[-1] + pd.DateOffset(months=1)] # Следующий месяц

    # SARIMAX Test
    if mode == "train_val":
        
        # Используем уже готовый y_test вместо повторного shift/loc
        y_test_sarimax = y_test
        
        # Берем прогнозы, сдвигаем их (чтобы выровнять прогноз на T с таргетом T-1) и выравниваем по индексу теста
        y_test_pred_sarimax = sarima_forecast.shift(-1).reindex(y_test.index)
        
        # ВАЖНО: Заполняем возможные NaN в прогнозе (если SARIMAX упал на последнем шаге)
        # Сначала пробуем заполнить предыдущим значением, если все равно NaN — нулем
        y_test_pred_sarimax = y_test_pred_sarimax.fillna(method='ffill').fillna(0)
        
        # Гарантируем, что индексы совпадают (хотя reindex уже это сделал)
        y_test_sarimax, y_test_pred_sarimax = y_test_sarimax.align(y_test_pred_sarimax, join="inner")
        
        # Метрики
        rmse_test_sarimax = np.sqrt(mean_squared_error(y_test_sarimax, y_test_pred_sarimax))
        mase_test_sarimax = mase(y_test_sarimax, y_test_pred_sarimax, pd.concat([y_train, y_val]), 12)
        
        
        #y_test_sarimax = data_fund.shift(-1).loc[data_fund.index >= test_start_date, "month_payments_sum"].copy().asfreq("MS")
        #y_test_pred_sarimax = sarima_forecast.shift(-1).loc[sarima_forecast.index >= test_start_date].copy().asfreq("MS")
        #y_test_sarimax, y_test_pred_sarimax = y_test_sarimax.align(y_test_pred_sarimax, join="inner")
        
        # Метрики
        #rmse_test_lstm = np.sqrt(mean_squared_error(y_test, y_test_pred_lstm))
        #rmse_test_sarimax = np.sqrt(mean_squared_error(y_test_sarimax, y_test_pred_sarimax))

    elif mode == "production":
        # Прогноз на 1 шаг вперед от последней обученной модели
        try:
             # Мы сохранили last_sarimax_results в цикле
             pred_sarima = last_sarimax_results.get_forecast(steps=1).predicted_mean.item()
        except:
             pred_sarima = sarima_forecast.iloc[-1] # fallback

        y_test_pred_sarimax = max(pred_sarima, 0)

    # Catboost Test
    X_test["sarima_forecast"] = sarima_forecast.shift(-1).reindex(X_test.index)
    X_test["sarima_forecast"] = X_test["sarima_forecast"].fillna(y_test_pred_sarimax if mode=='production' else 0)

    y_test_pred_cb = np.clip(np.round(final_model_cb.predict(X_test), decimals=2), 0, None)
    
    if mode == "production":
        y_test_pred_cb = float(y_test_pred_cb[0])

    # %%
    # Сборка
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
        ).fillna(0)
    
    elif mode == "production":
        predict_date = y_test_idx[0]
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
    # Ансамбль predict
    X_meta_test = forecasts_test[["forecast_lstm", "forecast_sarimax", "forecast_catboost"]]
    
    # Ridge
    if mode == "production":
        forecasts_test["forecast_ridge"] = np.round(np.clip(ridge_model.predict(X_meta_test).item(), 0, None), 2)
    else:
        forecasts_test["forecast_ridge"] = np.round(np.clip(ridge_model.predict(X_meta_test), 0, None), 2)

    # NN
    X_meta_test_tensor = torch.tensor(X_meta_test.values, dtype=torch.float32)
    if mode == "production":
        forecasts_test["forecast_nn"] = np.round(np.clip(model_meta(X_meta_test_tensor).detach().numpy().item(), 0, None),2)
    else:
        forecasts_test["forecast_nn"] = np.round(np.clip(model_meta(X_meta_test_tensor).detach().numpy(), 0, None),2)

    # %%
    # Сохранение результатов
    if mode == "train_val":
        # История для расчета MASE (Train + Val)
        history_series = pd.concat([y_train, y_val])

        # Метрики для NN
        mase_nn = mase(forecasts_test["y_actual"], forecasts_test["forecast_nn"], history_series, 12)
        rmse_nn = np.sqrt(mean_squared_error(forecasts_test["y_actual"], forecasts_test["forecast_nn"]))

        # --- ДОБАВЛЕНО: Метрики для Ridge ---
        mase_ridge = mase(forecasts_test["y_actual"], forecasts_test["forecast_ridge"], history_series, 12)
        rmse_ridge = np.sqrt(mean_squared_error(forecasts_test["y_actual"], forecasts_test["forecast_ridge"]))

        path3 = "comparison_train_val_monthly.csv"
        if os.path.exists(path3):
            comparison = pd.read_csv(path3)
        else:
            comparison = pd.DataFrame(columns=["fund_id"])

        # Обновляем или добавляем строку
        if FUND_ID in comparison["fund_id"].values:
            idx = comparison[comparison["fund_id"] == FUND_ID].index
            comparison.loc[idx, "3models_ens_nn_rmse"] = rmse_nn
            comparison.loc[idx, "3models_ens_nn_mase"] = mase_nn
            # Добавляем колонки Ridge
            comparison.loc[idx, "3models_ens_ridge_rmse"] = rmse_ridge
            comparison.loc[idx, "3models_ens_ridge_mase"] = mase_ridge
        else:
            new_row = pd.DataFrame({
                "fund_id": [FUND_ID],
                "3models_ens_nn_rmse": [rmse_nn],
                "3models_ens_nn_mase": [mase_nn],
                "3models_ens_ridge_rmse": [rmse_ridge],
                "3models_ens_ridge_mase": [mase_ridge],
            })
            comparison = pd.concat([comparison, new_row], ignore_index=True)
        comparison.to_csv(path3, index=False)

    elif mode == "production":
        path4 = "forecasts_full_monthly.csv"
        file_exists = os.path.isfile(path4)
        forecasts_test.to_csv(path4, mode="a", header=not file_exists, index=True)

# %%
# Медианный прогноз для остальных
if mode == 'production':
    forecast_date = predict_date.strftime('%Y-%m-%d')
    
    if os.path.exists(path4):
        forecasts_df = pd.read_csv(path4, index_col=0)
        forecast_next = (forecasts_df[forecasts_df.index == forecast_date][['fund_id', 'forecast_nn']]
                            .rename(columns={'forecast_nn': 'forecast'}))

        other_funds = list(set(data_final['user_id'].unique()) - set(forecast_next['fund_id'].unique()))
        median_rows = []

        for o_fund in other_funds:
            median_val = data_final[data_final['user_id'] == o_fund]['month_payments_sum'].median()
            median_rows.append({
                'fund_id': o_fund,
                'forecast': round(median_val, 2)
            })
        
        median_forecast_df = pd.DataFrame(median_rows)
        if not median_forecast_df.empty:
            median_forecast_df.index = [forecast_date] * len(median_forecast_df)
            forecast_full = pd.concat([forecast_next, median_forecast_df], axis=0)
        else:
            forecast_full = forecast_next

        forecast_full = forecast_full.reset_index().sort_values(by=['index', 'fund_id']).set_index('index')
        forecast_full.index.name = None

        path5 = 'forecast_next_month.csv'
        if os.path.exists(path5):
            df_old = pd.read_csv(path5, index_col=0)
            df = pd.concat([df_old, forecast_full], axis=0)
        else:
            df = forecast_full

        df.to_csv(path5)
        logger.info(f'ℹ️ Данные прогнозов сохранены в файл {path5}')

logger.info(f"✅ Завершение скрипта")