# %%
# ## Паплайн для загрузки данных, предобработки, генерации прогнозов и записи в БД

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


import pandas as pd
import numpy as np
import random
import os
import re
import requests
import joblib
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA

# ⏳ прогресс-бары
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

# загрузка catboost
from catboost import CatBoostClassifier


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 0)
pd.set_option('display.max_colwidth', 120)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# %%
# функция очистки текста
def _clean_single_text(text):
    return re.sub(r"[^\w\s]", " ", text.lower())

# функция предобработки текста
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

# функция для получения усредненного эмбеддинга текста
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

# функция предобработки входного датасета в целом
def prepare_data(data, is_train, scaler=None, ipca=None):
    
    data = data.drop(
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
            ],
            axis=1,
        )

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
    data["articles__name"] = data["articles__name"].fillna("missing")
    data["projects__name"] = data["projects__name"].fillna("missing")
    data["counterparties__name"] = data["counterparties__name"].fillna("missing")

    # конвертируем дату в datetime
    data["date"] = pd.to_datetime(data["date"], yearfirst=True, errors='coerce')

    # и убираем записи из будущего и меньше нуля (и такое бывает)
    yesterday = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    data = data[data["date"] <= yesterday]
    #data = data[data["payments_amount"] > 0]

    # переименуем и поправим тип столбца с фондами
    data = data.rename(columns={"accounts__user_id": "user_id"})
    data["user_id"] = data["user_id"].fillna(0).astype("int64")

    # кодируем текстовые поля
    # сначала очищаем и лемматизируем тексты
    data["clean_purpose"] = preprocess_texts_optimized(texts=data["purpose"],nlp_model_name="ru_core_news_sm")

    # грузим модели 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")
    model = model.to(device)
    model.eval()
    
    # и запускаем генерацию эмбеддингов в назначении платежа
    data["purpose_emb"] = get_embeddings_batch(data["clean_purpose"], tokenizer, model, device)

    # 1. усредняем эмбеддинг в одно число
    data["purpose_mean"] = data["purpose_emb"].apply(lambda x: float(np.mean(x)))

    # 2. выделяем три главные компоненты с предварительным масштабированием и по батчам  
    batch_size = 10_000
    
    if is_train:
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
    for i in tqdm(range(0, len(data), batch_size), desc="Применяем PCA к эмбеддингам"):
        batch = np.vstack(data["purpose_emb"].iloc[i:i+batch_size]).astype(np.float32)
        batch_scaled = scaler.transform(batch)
        transformed_batches.append(ipca.transform(batch_scaled))
        
    purpose_pca_features = np.vstack(transformed_batches)

    # делаем датафрейм
    pca_column_names = [f"purpose_pca_{i+1}" for i in range(3)]
    data[pca_column_names] = purpose_pca_features

    # удалим ненужные столбцы
    data.drop(columns=["purpose", "clean_purpose", "purpose_emb"], inplace=True)

    # генерируем эмбеддинги для названий статей
    # сначала очищаем и лемматизируем тексты
    data["clean_articles__name"] = preprocess_texts_optimized(texts=data["articles__name"],nlp_model_name="ru_core_news_sm")
     # и запускаем генерацию эмбеддингов в названии статей
    data["articles__name_emb"] = get_embeddings_batch(data["clean_articles__name"], tokenizer, model, device)
    # усредняем эмбеддинг в одно число
    data["articles__name_mean"] = data["articles__name_emb"].apply(lambda x: float(np.mean(x)))
    # удалим ненужные столбцы
    data.drop(columns=["articles__name", "clean_articles__name", "articles__name_emb"], inplace=True)

    # генерируем эмбеддинги для названий проектов
    # сначала очищаем и лемматизируем тексты
    data["clean_projects__name"] = preprocess_texts_optimized(texts=data["projects__name"],nlp_model_name="ru_core_news_sm")
     # и запускаем генерацию эмбеддингов в названии статей
    data["projects__name_emb"] = get_embeddings_batch(data["clean_projects__name"], tokenizer, model, device)
    # усредняем эмбеддинг в одно число
    data["projects__name_mean"] = data["projects__name_emb"].apply(lambda x: float(np.mean(x)))
    # удалим ненужные столбцы
    data.drop(columns=["projects__name", "clean_projects__name", "projects__name_emb"], inplace=True)
    
    # генерируем эмбеддинги для названий доноров
    # сначала очищаем и лемматизируем тексты
    data["clean_counterparties__name"] = preprocess_texts_optimized(texts=data["counterparties__name"],nlp_model_name="ru_core_news_sm")
     # и запускаем генерацию эмбеддингов в названии статей
    data["counterparties__name_emb"] = get_embeddings_batch(data["clean_counterparties__name"], tokenizer, model, device)
    # усредняем эмбеддинг в одно число
    data["counterparties__name_mean"] = data["counterparties__name_emb"].apply(lambda x: float(np.mean(x)))
    # удалим ненужные столбцы
    data.drop(columns=["counterparties__name", "clean_counterparties__name", "counterparties__name_emb"], inplace=True)


    # сбрасываем неактуальные столбцы
    data = data.drop(columns=[ 
        'date', 'expenditure',
        'payments_amount','user_id','account_id', 
        'contractor_id', 'article_id', 'project_id', 
        'counterpartie_id', 'donor_id', 'donor_cat_id',
        'articles__parent_id', 'projects__parent_id',
        'counterparties__parent_id', 'robots__id'])

    return data,scaler,ipca

# %%
# настраиваем логгер
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# файл
file_handler = logging.FileHandler("classify_forecast_prod.log")
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

# добавляем обработчики в логгер
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# %%

if __name__ == "__main__":

    logger.info(f"✅ Запуск скрипта")

    # %%
    # загружаем новые платежи за вчера или все, если воскресенье

    url_down = "https://api.lemonpie.tech/api/payments/ai"
    headers = {"Authorization": "Bearer YOUR_API_TOKEN"}

    today = pd.Timestamp.today().normalize()
    weekday = today.weekday()

    if weekday == 6:  # в воскресенье выкачиваем все данные и дозаполняем метками uc
        start_date = ""
        end_date = ""
    else:
        start_date = str((today - pd.Timedelta(days=1)).date())
        end_date = start_date


    params = {
        "limit": 5000,
        "page": 1,
        "expenditure": "incoming",
        "date_from": start_date,
        "date_to": end_date  
    }

    all_data = []

    with tqdm(desc="Загружено страниц", unit=" стр", dynamic_ncols=True) as pbar:
        while True:
            response = requests.get(url_down, headers=headers, params=params)
            if response.status_code != 200:
                logger.info(f"❌ Ошибка загрузки данных с сервера: {response.status_code}")
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
    logger.info(f"Данные загружены с сервера. Количество записей: {len(data_full)}")
    #data_full.to_csv("data_download.csv", index=False)

    # %%
    logger.debug(
        f"Проверим пропуски по основным признакам:\n"
        f"{data_full[['purpose','articles__name','projects__name','counterparties__name']].isna().sum()}"
    )

    logger.debug(
        f"Количество строк, где все 3 дополнительных признака отсутствуют: "
        f"{data_full[['articles__name','projects__name','counterparties__name']].isna().all(axis=1).sum()}"
    )

    # сбросим строки в которых все 3 дополнительных поля отсутствуют (роботы не отработали - качество прогноза будет плохое)
    data_full = data_full.dropna(subset=['articles__name', 'projects__name', 'counterparties__name'], how='all')
    # или в которых уже есть метки uc
    data_full = data_full[data_full['uc__uc_id'].isna()]

    # %%
    # предобрабатываем датасет
    scaler_emb_path = 'scaler.pkl'
    ipca_emb_path = 'ipca.pkl'

    scaler = joblib.load(scaler_emb_path)
    ipca = joblib.load(ipca_emb_path)

    data_full_prepared, _, _ = prepare_data(data_full, is_train=False, scaler=scaler, ipca=ipca)

    # %%
    # ### генерируем прогнозы/catboost

    best_model_cb = CatBoostClassifier()
    best_model_cb.load_model("model_cb.cbm")
    logger.debug(f"Параметры загруженной модели:\n{best_model_cb.get_params()}")

    features = [
        'purpose_mean', 'purpose_pca_1', 'purpose_pca_2', 'purpose_pca_3',
        'articles__name_mean', 'projects__name_mean', 'counterparties__name_mean'
    ]

    data_full_prepared['uc__uc_id'] = best_model_cb.predict(data_full_prepared[features])
    logger.info(f"Прогнозы сгенерированы")


    # %%
    # генерируем запрос на запись в БД
    payload = {
        "items": [
            {"payment_id": int(row.id), "uc_id": int(row.uc__uc_id)}
            for row in data_full_prepared.itertuples(index=False)
        ]
    }

    # %%
    # отправляем запрос на запись в БД
    url_up = "https://api.lemonpie.tech/api/payments/article-u"
    headers = {
        "Authorization": "Bearer YOUR_API_TOKEN",
        "Content-Type": "application/json",
    }

    response = requests.post(url_up, json=payload, headers=headers)
    logger.info(f"Статус загрузки в БД: {response.status_code}, {response.text}")
    logger.info(f"✅ Завершение скрипта")
