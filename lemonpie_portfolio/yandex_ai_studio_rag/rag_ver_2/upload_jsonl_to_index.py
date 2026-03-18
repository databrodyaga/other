import sys
import pathlib
import json
import time
from openai import OpenAI


YANDEX_API_KEY = ""
YANDEX_FOLDER_ID = "YOUR_FOLDER_ID"

DATA_DIR = pathlib.Path(__file__).parent
INDEX_NAME = "База знаний LemonPie"


def is_valid_jsonl_body_schema(path: pathlib.Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    return False
                if "body" not in obj or not isinstance(obj["body"], str):
                    return False
        return True
    except Exception:
        return False


def select_jsonl_files_not_uploaded(client):
    local = {
        p.name: p
        for p in DATA_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".jsonl"
    }

    server_files = client.files.list().data
    server_names = {pathlib.Path(f.filename).name for f in server_files}

    return [
        (name, path)
        for name, path in local.items()
        if name not in server_names
    ]


def select_index(client):
    stores = client.vector_stores.list().data
    for s in stores:
        if s.name == INDEX_NAME:
            return s.id
    raise RuntimeError(f"Индекс '{INDEX_NAME}' не найден")


def upload_and_index_jsonl(client, index_id):
    candidates = select_jsonl_files_not_uploaded(client)

    if not candidates:
        print("Новых jsonl-файлов нет.")
        return

    print("Доступные jsonl файлы:")
    for i, (name, _) in enumerate(candidates, 1):
        print(f"{i}. {name}")

    choice = input("Выберите файл для загрузки: ").strip()
    if not choice:
        print("Отменено.")
        return

    try:
        idx = int(choice) - 1
        name, path = candidates[idx]
    except Exception:
        print("Некорректный выбор.")
        return

    print(f"\nПроверяю схему файла: {name}")
    if not is_valid_jsonl_body_schema(path):
        print("❌ Некорректная схема JSONL (ожидается {'body': str})")
        return

    print("✅ Схема корректна. Загружаю файл...")

    with open(path, "rb") as f:
        file_resp = client.files.create(
            file=(name, f, "application/jsonlines"),
            purpose="assistants",
            extra_body={"format": "chunks"},
        )

    print(f"Файл загружен на сервер: id={file_resp.id}")

    batch = client.vector_stores.file_batches.create(
        vector_store_id=index_id,
        file_ids=[file_resp.id],
    )

    while True:
        batch = client.vector_stores.file_batches.retrieve(
            vector_store_id=index_id,
            batch_id=batch.id,
        )

        if batch.status == "in_progress":
            sys.stdout.write("\rДобавляю файл в индекс...")
            sys.stdout.flush()
            time.sleep(2)
            continue

        if batch.status == "completed":
            break

        raise RuntimeError(f"Ошибка индексации: {batch.status}")

    print("\n✅ Файл успешно добавлен в индекс")


if __name__ == "__main__":
    client = OpenAI(
        api_key=YANDEX_API_KEY,
        base_url="https://rest-assistant.api.cloud.yandex.net/v1",
        project=YANDEX_FOLDER_ID,
    )

    index_id = select_index(client)
    upload_and_index_jsonl(client, index_id)