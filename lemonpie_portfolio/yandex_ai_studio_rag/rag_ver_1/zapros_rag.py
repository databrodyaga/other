from yandex_cloud_ml_sdk import YCloudML #pip install yandex-cloud-ml-sdk

folder = "YOUR_FOLDER_ID"
token = "" # мой токенот сервисного аккаунта к yandex cloud ai
assistant_id = "fvtitjq11gdjlssc81tc" # на базе qwen3-235b или fvtlcenkrulgtgd6boko - yandexgpt pro

def main():
    sdk = YCloudML(folder_id=folder, auth=token)

    assistant = sdk.assistants.get(assistant_id)
    thread = sdk.threads.create()

    while True:
        q = input("Ваш вопрос: ")
        if q.lower() == "exit":
            break

        thread.write(q)

        run = assistant.run(thread)

        result = run.wait()

        print("Ответ:", result.text)

    thread.delete()

if __name__ == "__main__":
    main()