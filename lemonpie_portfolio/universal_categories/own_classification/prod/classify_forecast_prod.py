# %% 
# Паплайн для загрузки данных, предобработки, генерации прогнозов и записи в БД

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
from datetime import date

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
DATA_DIR = os.getenv("DATA_DIR", ".")


# %%
STATE_PATH = os.path.join(DATA_DIR, "last_id.txt")

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# %%
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

# функция предобработки входного датасета
def prepare_data(data, is_train, scaler=None, ipca=None, 
                                 scaler_art=None, ipca_art=None,
                                 scaler_pro=None, ipca_pro=None,
                                 scaler_cou=None, ipca_cou=None):
    
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
    for i in tqdm(range(0, len(data), batch_size), desc="Применяем PCA к purpose"):
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
    # выделяем 3-PCA, полностью аналогично purpose
    batch_size = 10_000
    if is_train:
        scaler_art = StandardScaler()
        ipca_art = IncrementalPCA(n_components=3)

        # обучаем scaler
        for i in tqdm(range(0, len(data), batch_size), desc="Обучение StandardScaler (articles)"):
            batch = np.vstack(data["articles__name_emb"].iloc[i:i+batch_size])
            scaler_art.partial_fit(batch)

        # обучаем PCA
        for i in tqdm(range(0, len(data), batch_size), desc="Обучение IncrementalPCA (articles)"):
            batch = np.vstack(data["articles__name_emb"].iloc[i:i+batch_size])
            batch_scaled = scaler_art.transform(batch)
            ipca_art.partial_fit(batch_scaled)

    # применяем трансформацию ко всем данным
    transformed_batches_art = []
    for i in tqdm(range(0, len(data), batch_size), desc="Применяем PCA к articles embeddings"):
        batch = np.vstack(data["articles__name_emb"].iloc[i:i+batch_size]).astype(np.float32)
        batch_scaled = scaler_art.transform(batch)
        transformed_batches_art.append(ipca_art.transform(batch_scaled))

    articles_pca_features = np.vstack(transformed_batches_art)

    # создаём колонки
    art_pca_colnames = [f"articles_pca_{i+1}" for i in range(3)]
    data[art_pca_colnames] = articles_pca_features
    # удалим ненужные столбцы
    data.drop(columns=["articles__name", "clean_articles__name", "articles__name_emb"], inplace=True)

    # генерируем эмбеддинги для названий проектов
    # сначала очищаем и лемматизируем тексты
    data["clean_projects__name"] = preprocess_texts_optimized(texts=data["projects__name"],nlp_model_name="ru_core_news_sm")
     # и запускаем генерацию эмбеддингов в названии статей
    data["projects__name_emb"] = get_embeddings_batch(data["clean_projects__name"], tokenizer, model, device)
    # усредняем эмбеддинг в одно число
    data["projects__name_mean"] = data["projects__name_emb"].apply(lambda x: float(np.mean(x)))
    
    # выделяем 3-PCA, полностью аналогично purpose
    batch_size = 10_000
    if is_train:
        scaler_pro = StandardScaler()
        ipca_pro = IncrementalPCA(n_components=3)

        # обучаем scaler
        for i in tqdm(range(0, len(data), batch_size), desc="Обучение StandardScaler (projects)"):
            batch = np.vstack(data["projects__name_emb"].iloc[i:i+batch_size])
            scaler_pro.partial_fit(batch)

        # обучаем PCA
        for i in tqdm(range(0, len(data), batch_size), desc="Обучение IncrementalPCA (projects)"):
            batch = np.vstack(data["projects__name_emb"].iloc[i:i+batch_size])
            batch_scaled = scaler_pro.transform(batch)
            ipca_pro.partial_fit(batch_scaled)

    # применяем трансформацию ко всем данным
    transformed_batches_pro = []
    for i in tqdm(range(0, len(data), batch_size), desc="Применяем PCA к projects embeddings"):
        batch = np.vstack(data["projects__name_emb"].iloc[i:i+batch_size]).astype(np.float32)
        batch_scaled = scaler_pro.transform(batch)
        transformed_batches_pro.append(ipca_pro.transform(batch_scaled))

    projects_pca_features = np.vstack(transformed_batches_pro)

    # создаём колонки
    pro_pca_colnames = [f"projects_pca_{i+1}" for i in range(3)]
    data[pro_pca_colnames] = projects_pca_features
    # удалим ненужные столбцы
    data.drop(columns=["projects__name", "clean_projects__name", "projects__name_emb"], inplace=True)
    
    # генерируем эмбеддинги для названий доноров
    # сначала очищаем и лемматизируем тексты
    data["clean_counterparties__name"] = preprocess_texts_optimized(texts=data["counterparties__name"],nlp_model_name="ru_core_news_sm")
     # и запускаем генерацию эмбеддингов в названии статей
    data["counterparties__name_emb"] = get_embeddings_batch(data["clean_counterparties__name"], tokenizer, model, device)
    # усредняем эмбеддинг в одно число
    data["counterparties__name_mean"] = data["counterparties__name_emb"].apply(lambda x: float(np.mean(x)))
    
    # выделяем 3-PCA, полностью аналогично purpose
    batch_size = 10_000

    if is_train:
        scaler_cou = StandardScaler()
        ipca_cou = IncrementalPCA(n_components=3)

        # обучаем scaler
        for i in tqdm(range(0, len(data), batch_size), desc="Обучение StandardScaler (counterparties)"):
            batch = np.vstack(data["counterparties__name_emb"].iloc[i:i+batch_size])
            scaler_cou.partial_fit(batch)

        # обучаем PCA
        for i in tqdm(range(0, len(data), batch_size), desc="Обучение IncrementalPCA (counterparties)"):
            batch = np.vstack(data["counterparties__name_emb"].iloc[i:i+batch_size])
            batch_scaled = scaler_cou.transform(batch)
            ipca_cou.partial_fit(batch_scaled)

    # применяем трансформацию ко всем данным
    transformed_batches_cou = []
    for i in tqdm(range(0, len(data), batch_size), desc="Применяем PCA к counterparties embeddings"):
        batch = np.vstack(data["counterparties__name_emb"].iloc[i:i+batch_size]).astype(np.float32)
        batch_scaled = scaler_cou.transform(batch)
        transformed_batches_cou.append(ipca_cou.transform(batch_scaled))

    counterparties_pca_features = np.vstack(transformed_batches_cou)

    # создаём колонки
    cou_pca_colnames = [f"counterparties_pca_{i+1}" for i in range(3)]
    data[cou_pca_colnames] = counterparties_pca_features
    # удалим ненужные столбцы
    data.drop(columns=["counterparties__name", "clean_counterparties__name", "counterparties__name_emb"], inplace=True)

    # сбрасываем неактуальные столбцы всего датасета
    data = data.drop(columns=[ 
        'date', 'expenditure',
        'payments_amount','user_id','account_id', 
        'contractor_id', 'article_id', 'project_id', 
        'counterpartie_id', 'donor_id', 'donor_cat_id',
        'articles__parent_id', 'projects__parent_id',
        'counterparties__parent_id', 'robots__id'])

    return data,scaler,ipca,scaler_art,ipca_art,scaler_pro,ipca_pro,scaler_cou,ipca_cou

def load_last_id():
    if not os.path.exists(STATE_PATH):
        return 0
    with open(STATE_PATH) as f:
        return int(f.read())

def save_last_id(last_id):
    with open(STATE_PATH, "w") as f:
        f.write(str(last_id))


# %%
# настраиваем логгер
LOG_PATH = os.path.join(DATA_DIR, "classify_forecast_prod.log")
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# файл
file_handler = logging.FileHandler(LOG_PATH)
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
    # Загрузим и подготовим данные для прогноза

    # настроим доступы по API

    # скачивание данных (read)
    API_URL_DOWN = os.getenv(
        "PAYMENTS_API_URL_DOWN",
        "https://api.lemonpie.tech/api/payments/ai"
    )
    API_TOKEN_DOWN = os.getenv("PAYMENTS_API_TOKEN_DOWN")

    # загрузка меток (write)
    API_URL_UP = os.getenv(
        "PAYMENTS_API_URL_UP",
        "https://api.lemonpie.tech/api/payments/article-u"
    )
    API_TOKEN_UP = os.getenv("PAYMENTS_API_TOKEN_UP")

    if not API_TOKEN_DOWN:
        raise RuntimeError("PAYMENTS_API_TOKEN_DOWN is not set")

    if not API_TOKEN_UP:
        raise RuntimeError("PAYMENTS_API_TOKEN_UP is not set")

    headers_down = {
        "Authorization": f"Bearer {API_TOKEN_DOWN}"
    }

    headers_up = {
        "Authorization": f"Bearer {API_TOKEN_UP}",
        "Content-Type": "application/json",
    }

    logger.info(f"GET-запрос по адресу {API_URL_DOWN}")
    logger.info(f"POST-запрос по адресу {API_URL_UP}")

    # %%
    # загружаем новые платежи в зависимости от режима запуска
    LOAD_MODE = os.getenv("LOAD_MODE", "incremental") # incremental или full или date
    LOAD_MODE = LOAD_MODE.lower()
    if LOAD_MODE not in {"incremental", "full", "date"}:
        raise ValueError(f"Неизвестный режим запуска LOAD_MODE: {LOAD_MODE}")
    logger.info(f"Режим запуска: {LOAD_MODE}")

    params = {
        "limit": 5000,
        "page": 1,
        "expenditure": "incoming",
    }

    if LOAD_MODE == "incremental":
        last_id = load_last_id()
        if last_id > 0:
            params["after_id"] = last_id

    elif LOAD_MODE == "full":
        pass 

    elif LOAD_MODE == "date":
        run_date_from = os.getenv("RUN_DATE_FROM")
        run_date_to = os.getenv("RUN_DATE_TO")

        # добавим проверки для дат
        if not run_date_from or not run_date_to:
            raise RuntimeError("LOAD_MODE=date требует параметров RUN_DATE_FROM и RUN_DATE_TO")
        from datetime import datetime
        
        try:
            datetime.strptime(run_date_from, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"RUN_DATE_FROM должен быть валидной датой YYYY-MM-DD, получено: {run_date_from}")

        try:
            datetime.strptime(run_date_to, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"RUN_DATE_TO должен быть валидной датой YYYY-MM-DD, получено: {run_date_to}")
        
        if run_date_from > run_date_to:
            raise ValueError(f"RUN_DATE_FROM ({run_date_from}) не может быть больше RUN_DATE_TO ({run_date_to})")

        params["date_from"] = run_date_from
        params["date_to"] = run_date_to

        logger.info(f"Режим date: диапазон дат {run_date_from} — {run_date_to}")

                
    all_data = []

    with tqdm(desc="Загружено страниц", unit=" стр", dynamic_ncols=True) as pbar:
        while True:
            response = requests.get(API_URL_DOWN, headers=headers_down, params=params,timeout=30)
            if response.status_code != 200:
                logger.info(f"❌ Ошибка загрузки данных с сервера: {response.status_code}")
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

    if len(data_full) == 0:
        logger.info("❌ Нет новых данных для прогноза")
        raise SystemExit


    # %%
    # предобрабатываем датасет
    scaler_emb_path = 'scaler.pkl'
    ipca_emb_path = 'ipca.pkl'
    scaler_art_emb_path = 'scaler_art.pkl'
    ipca_art_emb_path = 'ipca_art.pkl'
    scaler_pro_emb_path = 'scaler_pro.pkl'
    ipca_pro_emb_path = 'ipca_pro.pkl'
    scaler_cou_emb_path = 'scaler_cou.pkl'
    ipca_cou_emb_path = 'ipca_cou.pkl'


    scaler = joblib.load(scaler_emb_path)
    ipca = joblib.load(ipca_emb_path)
    scaler_art = joblib.load(scaler_art_emb_path)
    ipca_art = joblib.load(ipca_art_emb_path)
    scaler_pro = joblib.load(scaler_pro_emb_path)
    ipca_pro = joblib.load(ipca_pro_emb_path)
    scaler_cou = joblib.load(scaler_cou_emb_path)
    ipca_cou = joblib.load(ipca_cou_emb_path)

    data_full_prepared,_,_,_,_,_,_,_,_ = prepare_data(data_full, is_train=False, scaler=scaler, ipca=ipca,
                                                                    scaler_art=scaler_art, ipca_art=ipca_art,
                                                                    scaler_pro=scaler_pro, ipca_pro=ipca_pro,
                                                                    scaler_cou=scaler_cou, ipca_cou=ipca_cou)

    # %%
    # генерируем прогнозы catboost

    # %%
    best_model_cb = CatBoostClassifier()
    best_model_cb.load_model("model_cb.cbm")
    logger.debug(f"Параметры загруженной модели:\n{best_model_cb.get_params()}")

    features = [
        'purpose_mean', 'purpose_pca_1', 'purpose_pca_2', 'purpose_pca_3',
        'articles__name_mean', 'articles_pca_1', 'articles_pca_2', 'articles_pca_3', 
        'projects__name_mean', 'projects_pca_1', 'projects_pca_2', 'projects_pca_3', 
        'counterparties__name_mean', 'counterparties_pca_1', 'counterparties_pca_2', 'counterparties_pca_3'
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
    response = requests.post(API_URL_UP, json=payload, headers=headers_up, timeout=30)

    if response.status_code != 200:
        logger.info(f"❌ Ошибка записи в БД: {response.status_code}, {response.text}")
        raise RuntimeError("❌ Ошибка записи в БД")

    logger.info(f"Статус загрузки в БД: {response.status_code}, {response.text}")

    # %%
    if LOAD_MODE == "incremental" and not data_full_prepared.empty:
        save_last_id(int(data_full_prepared["id"].max()))
        logger.info(f"ID последнего обработанного платежа {int(data_full_prepared['id'].max())} сохранен в файл")
    logger.info(f"✅ Завершение скрипта")
