import subprocess
import sys
import requests

try:
    import openai
    from openai import OpenAI
except ImportError:
    print("Библиотека openai не найдена. Устанавливаю...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    import openai
    from openai import OpenAI


# история диалога c системным промптом
history = [
    {
        "role": "system",
        "content": (
            "Ты — ассистент технической поддержки. Ты общаешься с пользователем системы и помогаешь решить его технические вопросы, а также рассказываешь про систему "
            "в целом из открытых источников. "
            "!СТРОГО - Если вопрос не относится к технической поддержке или продукту LemonPie(Лемонпай), вежливо откажись отвечать!"
            "Для получения информации используй инструмент поиска по документации и отвечай "
            "на основе предоставленной базы знаний. "
            "Если в базе знаний нет информации для ответа, воспользуйся инструментом websearch. "
            "Если и так не сможешь сформулировать ответ, прямо скажи пользователю: "
            "«К сожалению, в базе знаний нет ответа на этот вопрос. Я могу передать запрос специалисту». "
            "Отвечай кратко, понятно и по делу. "
            "Если ты используешь информацию из базы знаний, в конце ответа напиши:'Источник: база знаний Лемонпай'"
            "Если ты используешь информацию из сети Интернет, в конце ответа напиши:'Источник: Интернет'"
            "Если отвечаешь с использованием обращения и к базе знаний и к сети Интернет, в конце ответа напиши:'Источник: база знаний Лемонпай, Интернет'"
            "Если отвечаешь без обращения к базе знаний или к сети Интернет, в конце ответа напиши:'Источник: общие знания модели'"
        )
    }
]

def select_vector_store(client):
    stores = client.vector_stores.list().data

    if not stores:
        print("Поисковый индекс не найден.")
        return None

    if len(stores) == 1:
        store = stores[0]
        print(f"Используется единственный индекс: {store.name} ({store.id})")
        return store.id

    print("Доступные индексы:")
    for i, s in enumerate(stores, 1):
        files = s.file_counts.total if s.file_counts else 0
        print(f"{i}. {s.name or '(без имени)'} | files={files} | id={s.id}")

    choice = input("Выберите номер индекса: ").strip()
    idx = int(choice) - 1
    return stores[idx].id

def ask_llm(client, question: str) -> str:
    history.append({"role": "user", "content": question})

    tools = []

    if VECTOR_STORE_ID:
        tools.append({
            "type": "file_search",
            "vector_store_ids": [VECTOR_STORE_ID],
        })

    tools.append({
        "type": "web_search",
        "web_search": {
            "filters": {
                "allowed_domains": []
            }
        }
    })

    # запрос
    response = client.responses.create(
    model=f"gpt://{FOLDER_ID}/{MODEL}",
    instructions=None,
    tools=tools,
    input=history,
    temperature=0.2,
    max_output_tokens=6000
    )

    answer = response.output_text
    history.append({"role": "assistant", "content": answer})
    return answer

def check_tools(client):
    print("\n=== Проверка инструментов ===")

    # 1. проверка поисковго индекса
    print("\n[VECTOR_STORE] Проверяю доступность индекса...")

    try:
        store = client.vector_stores.retrieve(VECTOR_STORE_ID)
    except Exception as e:
        print("✖ Не удалось получить Vector Store:", e)
        return

    print("✔ Vector Store доступен")
    print("ID:", store.id)
    print("Название:", store.name)
    print("Статус:", store.status)

    files_count = store.file_counts.total if store.file_counts else 0
    print("Файлов в индексе:", files_count)

    # 2. проверка файлов в индексе
    print("\nФайлы в индексе:")

    vs_files = client.vector_stores.files.list(vector_store_id=VECTOR_STORE_ID)

    if not vs_files.data:
        print("  (файлов нет)")
    else:
        for f in vs_files.data:
            file = client.files.retrieve(f.id)
            print(f" • {file.filename} | id={file.id}")

    # 3. проверка поиска
    print("\n[FILE_SEARCH] Проверяю поиск по индексу...")

    test_query = "донор"

    try:
        results = client.vector_stores.search(
            VECTOR_STORE_ID,
            query=test_query,
        )
    except Exception as e:
        print("✖ Ошибка при поиске:", e)
        return

    results_list = list(results)

    if results_list:
        print("✔ Поиск работает, найдено результатов:", len(results_list))
        print('Первый результат: ', results_list[0].content)
    else:
        print("✔ Поиск работает, но результатов нет (индекс пуст или нет совпадений)")
    
    # 4. проверка поиска по сети
    
    print("\n[WEB_SEARCH] Проверяю доступность web_search...")

    response = client.responses.create(
        model=f"gpt://{FOLDER_ID}/{MODEL}",

        input="Какая погода в Москве сегодня?",

        tools=[
            {
                "type": "web_search",
                "web_search": {
                    "filters": {
                        "allowed_domains": []
                    },
                    "user_location": {
                        "region": "213"
                    }
                }
            }
        ],

        temperature=0.0,
        max_output_tokens=1000
    )

    print("Ответ получен:", response.output_text)
    print("\n=== Проверка завершена ===\n")    
                    

if __name__ == "__main__":
    # параметры
    FOLDER_ID = "YOUR_FOLDER_ID"
    API_KEY = ""
    MODEL = "qwen3-235b-a22b-fp8/latest" #MODEL = "yandexgpt/latest" 

    # запуск агента
    client = openai.OpenAI(
        api_key=API_KEY,
        base_url="https://rest-assistant.api.cloud.yandex.net/v1",
        project=FOLDER_ID
    )
    
    VECTOR_STORE_ID = select_vector_store(client)

    while True:
        q = input("Ваш вопрос: (или Enter для выхода): ")

        if q.lower() == "":
            break
        if q.lower() == "check":
            check_tools(client)
            continue

        answer = ask_llm(client, q)
        print("Ответ:", answer)