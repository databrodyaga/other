import json

input_file = "faq_qa_block1.jsonl"

# Считываем первые 5 строк JSONL
records = []
with open(input_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        records.append(json.loads(line))

# Создаём 5 Markdown файлов
for idx, item in enumerate(records, start=1):
    question = item.get("question", "").strip()
    answer = item.get("answer", "").strip()

    filename = f"faq_{idx}.md"

    content = f"**Вопрос:** {question}\n\n**Ответ:** {answer}\n"

    with open(filename, "w", encoding="utf-8") as out:
        out.write(content)

    print(f"Создан файл: {filename}")