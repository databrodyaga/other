import subprocess
import sys
import pathlib
import json
import time

try:
    import openai
    from openai import OpenAI
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    import openai
    from openai import OpenAI


YANDEX_API_KEY = ""
YANDEX_FOLDER_ID = "YOUR_FOLDER_ID"

# каталог с локальными файлами
DATA_DIR = pathlib.Path(__file__).parent

# проверяем индекс — создать индекс, если его нет
def ensure_index(client, index_name="База знаний LemonPie"):
    # 1. проверяем существование индекса
    stores = client.vector_stores.list()
    
    for store in stores.data:
        if store.name == index_name:
            print(
                f"Индекс уже существует:\n"
                f"  id: {store.id}\n"
                f"  status: {store.status}"
            )
            files = client.vector_stores.files.list(store.id)
            if not files.data:
                print("  Файлов в индексе нет.")
            else:
                print("  Файлы в индексе:")
                for f in files.data:
                    try:
                        meta = client.files.retrieve(f.id)
                        print(f"   • {meta.filename} | id={f.id}")
                    except Exception:
                        # если вдруг файл удалён на сервере
                        print(f"   • id={f.id} (имя недоступно)")

            return store.id

    print("Индекс не найден.")

    # 2. получаем файлы на сервере
    files = client.files.list().data

    if not files:
        print("На сервере нет файлов. Создаю пустой индекс.")
        store = client.vector_stores.create(name=index_name)
        print(f"Индекс создан: {store.id}")
        return store.id

    # 3. показываем файлы
    print("\nФайлы на сервере:")
    for i, f in enumerate(files, start=1):
        print(f"{i}. {f.filename} | id={f.id}")

    raw = input("\nВведите номера файлов для индекса (через пробел/запятую, Enter — пустой индекс): ").strip()

    if not raw:
        store = client.vector_stores.create(name=index_name)
        print(f"Создан пустой индекс: {store.id}")
        return store.id

    try:
        idxs = {int(x) - 1 for x in raw.replace(",", " ").split()}
    except ValueError:
        print("Некорректный ввод. Создаю пустой индекс.")
        store = client.vector_stores.create(name=index_name)
        return store.id

    file_ids = [files[i].id for i in idxs if 0 <= i < len(files)]

    store = client.vector_stores.create(
        name=index_name,
        file_ids=file_ids,
    )

    print(f"Индекс создан: {store.id}")
    print(f"Файлов в индексе: {len(file_ids)}")
    return store.id

# проверка jsonl-файлов на формат схемы для чанков
def is_valid_jsonl_body_schema(path: pathlib.Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)

                if not isinstance(obj, dict):
                    return False
                if "body" not in obj:
                    return False
                if not isinstance(obj["body"], str):
                    return False

        return True

    except Exception:
        return False

# добавление файлов на сервер
def sync_files(client, index_id):
    # смотрим локальные файлы
    local_files = {
        p.name: p for p in DATA_DIR.iterdir() if p.is_file()
    }

    # смотрим файлы на сервере
    server_files = client.files.list()
    server_map = {
        pathlib.Path(f.filename).name: f.id
        for f in server_files.data
    }

    # находим отсутствующие файлы относительно локальных
    new_files = [
        (name, path)
        for name, path in local_files.items()
        if name not in server_map
    ]

    if not new_files:
        print("Новых файлов для загрузки нет.")
        return

    print("\nНовые файлы:")
    for i, (name, _) in enumerate(new_files, start=1):
        print(f"{i}. {name}")

    raw = input("\nВведите номера файлов для загрузки (через пробел или запятую, Enter — отмена): ").strip()

    if not raw:
        print("Отменено.")
        return

    try:
        idxs = {int(x) - 1 for x in raw.replace(",", " ").split()}
    except ValueError:
        print("Некорректный ввод.")
        return

    selected = [new_files[i] for i in idxs if 0 <= i < len(new_files)]

    if not selected:
        print("Файлы не выбраны.")
        return

    uploaded_file_ids = []

    EXT_TO_MIME = {
        ".json": "application/json",
        ".jsonl": "application/jsonlines",

        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".txt": "text/plain",
        ".csv": "text/csv",
        ".html": "text/html",
        ".xml": "text/xml",

        ".pdf": "application/pdf",
        ".rtf": "application/rtf",

        ".doc": "application/msword",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",

        ".xls": "application/vnd.ms-excel",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",

        ".ppt": "application/vnd.ms-powerpoint",

        ".tex": "application/x-latex",
        ".xhtml": "application/xhtml+xml",
    }

    print("\nЗагружаю файлы...\n")

    for name, path in selected:
        suffix = path.suffix.lower()

        with open(path, "rb") as f:

            # корректный jsonl → чанки
            if suffix == ".jsonl" and is_valid_jsonl_body_schema(path):
                resp = client.files.create(
                    file=(name, f, "application/jsonlines"),
                    purpose="assistants",
                    extra_body={"format": "chunks"},
                )
                print(f"→ JSONL chunks: {name}")

            else:
                mime = EXT_TO_MIME.get(suffix)

                if not mime:
                    print(f"Пропускаю неподдерживаемый файл: {name}")
                    continue

                resp = client.files.create(
                    file=(name, f, mime),
                    purpose="assistants",
                )

                if suffix == ".jsonl":
                    print(f"→ JSONL как обычный текст: {name}")
                else:
                    print(f"→ {mime}: {name}")

        uploaded_file_ids.append(resp.id)
        print(f"   загружен: {name} → id={resp.id}")

    if not uploaded_file_ids:
        print("Новых файлов нет — индекс не обновляется.")
        return

    # обновление индекса
    batch = client.vector_stores.file_batches.create(
        vector_store_id=index_id,
        file_ids=uploaded_file_ids,
    )

    while True:
        batch = client.vector_stores.file_batches.retrieve(
            vector_store_id=index_id,
            batch_id=batch.id,
        )
        if batch.status == "in_progress":
            sys.stdout.write("\rОбновляю индекс...")
            sys.stdout.flush()
            time.sleep(2)
            continue
        if batch.status == "completed":
            break
        if batch.status in ("failed", "cancelled"):
            raise RuntimeError(f"Обновление индекса завершилось ошибкой: {batch.status}")

    print("Индекс обновлён.")

# удаление файлов
def delete_files(client):
    files = client.files.list()

    if not files.data:
        print("На сервере нет файлов.")
        return

    print("Файлы на сервере:")
    for i, f in enumerate(files.data, start=1):
        print(f"{i}. {f.filename} | id={f.id}")

    raw = input("\nВведите номера файлов для удаления (через пробел или запятую, Enter — отмена): ").strip()

    if not raw:
        print("Отменено.")
        return

    try:
        idxs = {int(x) - 1 for x in raw.replace(",", " ").split()}
    except ValueError:
        print("Некорректный ввод.")
        return

    selected = [files.data[i] for i in idxs if 0 <= i < len(files.data)]

    if not selected:
        print("Файлы не выбраны.")
        return

    for f in selected:
        client.files.delete(f.id)
        print(f"Файл удалён: {f.filename}")

# удаление индекса
def delete_index(client):
    stores = client.vector_stores.list()

    if not stores.data:
        print("Нет доступных индексов.")
        return

    print("Доступные индексы:")
    for i, store in enumerate(stores.data, start=1):
        files_count = store.file_counts.total if store.file_counts else 0
        print(
            f"{i}. {store.name or '(без имени)'} | "
            f"id={store.id} | "
            f"files={files_count} | "
            f"status={store.status}"
        )

    choice = input("\nВведите номер индекса для удаления (или Enter для отмены): ").strip()
    if not choice:
        print("Отменено.")
        return

    idx = int(choice) - 1
    if idx < 0 or idx >= len(stores.data):
        print("Неверный номер индекса.")
        return
    store = stores.data[idx]

    if store.status == "in_progress":
        print("Индекс ещё создаётся. Удаление запрещено.")
        return

    client.vector_stores.delete(store.id)
    print(f"Индекс удален: {store.id}")


if __name__ == "__main__":
    client = OpenAI(
        api_key=YANDEX_API_KEY,
        base_url="https://rest-assistant.api.cloud.yandex.net/v1",
        project=YANDEX_FOLDER_ID,
    )

    while True:
        choice = input(
            "\nВыберите действие:\n"
            "1. Проверить/создать индекс\n"
            "2. Добавить локальные файлы в индекс\n"
            "3. Удалить файлы на сервере\n"
            "4. Удалить индекс\n"
            "Enter - выход\n"
            "> "
        ).strip().lower()

        if choice == "1":
            ensure_index(client)
        elif choice == "2":
            index_id = ensure_index(client)
            sync_files(client, index_id)
        elif choice == "3":
            delete_files(client)
        elif choice == "4":
            delete_index(client)
        elif choice == "":
            sys.exit(0)
        else:
            print("Неверный выбор.")