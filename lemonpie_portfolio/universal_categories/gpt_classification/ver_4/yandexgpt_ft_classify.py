import pandas as pd
from tqdm import tqdm
import numpy as np
import time

from __future__ import annotations
from yandex_cloud_ml_sdk import YCloudML

X_test = pd.read_csv('X_test_balanced.csv',index_col=0) # входные тексты назначений платежей

y_pred_ygpt_ft = pd.DataFrame(columns=["universal_category"]) # для прогнозов

sdk = YCloudML(
        folder_id="YOUR_FOLDER_ID",
        auth="YOUR_YANDEX_CLOUD_TOKEN",
    )

model = sdk.models.text_classifiers(
        "cls://YOUR_FOLDER_ID/yandexgpt-lite/rc@tamr9067f9ueb6v7f49gl"
    )

for index__, text in tqdm(X_test['purpose'].items(), total=len(X_test)):
    attempts = 0
    max_attempts = 3 # попытки связаться с сервером yandexgpt
    while attempts < max_attempts:
        try:
            result = model.run(str(text))
            best_label = max(result, key=lambda x: x.confidence).label
            y_pred_ygpt_ft.loc[index__, "universal_category"] = best_label
            break
        except Exception as e:
            attempts += 1
            print(f"Ошибка на index {index__}, попытка {attempts}/{max_attempts}: {e}")
            time.sleep(2)
    else:
        # если все попытки неудачные, присваиваем NaN
        y_pred_ygpt_ft.loc[index__, "universal_category"] = np.nan

        
y_pred_ygpt_ft.to_csv("y_pred_ygpt_ft.csv", index=True, header=["universal_category"])