
from __future__ import annotations
from yandex_cloud_ml_sdk import YCloudML

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
import time

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option("display.max_colwidth", None)


# %%
# звгружаем новые платежи за вчера

url_down = "https://api.lemonpie.tech/api/payments/ai"
headers = {"Authorization": ""}

start_date = str((pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).date())
end_date = str(pd.Timestamp.today().normalize().date())

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
            print(f"❌ Ошибка загрузки данных с сервера: {response.status_code}")
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
print(f"ℹ️ Данные загружены с сервера. Количество записей: {len(data_full)}")
#data_full.to_csv("data_download.csv", index=False)

# %%

print('Проверим пропуски по основным признакам:',
    (data_full[['purpose','articles__name','projects__name','counterparties__name']].isna().sum()))

print(f"Количество строк, где все 3 дополнительных признака отсутствуют: "
        f"{data_full[['articles__name','projects__name','counterparties__name']].isna().all(axis=1).sum()}")    

# сбросим строки в которых все 3 дополнительных поля отсутствуют (роботы не отработали - качество прогноза будет плохое)
data_full = data_full.dropna(subset=['articles__name', 'projects__name', 'counterparties__name'], how='all')

# или в которых уже есть метки uc
data_full = data_full[data_full['uc__uc_id'].isna()]

# и заполним пропуски, там где они еще останутся
data_full[['purpose','articles__name','projects__name','counterparties__name']] = data_full[['purpose','articles__name','projects__name','counterparties__name']].fillna('отсутствует') 

# %%
# переформатируем датасет для запроса к yandexgpt

zapros = data_full[['id','purpose','articles__name','projects__name','counterparties__name']].copy()

zapros.loc[:, 'text']  = zapros.apply(
    lambda x: (
        f"""назначение платежа: {x['purpose']}
название статьи: {x['articles__name']}
название проекта: {x['projects__name']}
категория донора: {x['counterparties__name']}"""
    ),
    axis=1
)

# %%
# отправляем запрос к yandexgpt
zapros["universal_category"] = None

sdk = YCloudML(
        folder_id="YOUR_FOLDER_ID",
        auth="",
    )

model = sdk.models.text_classifiers(
        "cls://YOUR_FOLDER_ID/yandexgpt-lite/latest@tamros8p69qq7ribmm06k"
    )

for index__, text in tqdm(zapros['text'].items(), total=len(zapros)):
    attempts = 0
    max_attempts = 3 # бывали сбои ответов, делаем три попытки
    while attempts < max_attempts:
        try:
            result = model.run(str(text))
            
            best_label = max(result, key=lambda x: x.confidence).label
            zapros.loc[index__, "universal_category"] = best_label
            break
        except Exception as e:
            attempts += 1
            print(f"Ошибка на index {index__}, попытка {attempts}/{max_attempts}: {e}")
            time.sleep(2)
    else:
        # если все попытки неудачные, присваиваем NaN
        zapros.loc[index__, "universal_category"] = np.nan


# %%
# кодируем категории uc согласованным словарем
uc_codex = {"пожертвования от физических лиц (напрямую)":1,
            "пожертвования через платформы":2,
            "пожертвования от юридических лиц (напрямую)":3,
            "прочие недоходные операции":4,
            "продажа услуг":5,
            "продажа товаров":6,
            "финансовые доходы":7,
            "членские и учредительские взносы":8,
            "гранты субсидии конкурсы":9}

zapros['uc_code'] = zapros['universal_category'].map(uc_codex)

print('Пропуски после кодирования:', zapros.uc_code.isna().sum()) # вдруг сервер где-то не ответил

# %%
# генерируем запрос на запись в БД
payload = {
    "items": [
        {"payment_id": int(row.id), "uc_id": int(row.uc_code)}
        for row in zapros.itertuples(index=False)
    ]
}

# %%
# отправляем запрос на запись в БД
url_up = "https://api.lemonpie.tech/api/payments/article-u"
headers = {
    "Authorization": "",
    "Content-Type": "application/json",
}

response = requests.post(url_up, json=payload, headers=headers)
print(response.status_code, response.text)


