import json
import pathlib

INPUT_FILE = pathlib.Path("faq_qa_block1.jsonl")
OUTPUT_FILE = pathlib.Path("faq_qa_block1_chunks.jsonl")

def convert_jsonl_to_chunks(src: pathlib.Path, dst: pathlib.Path):
    with src.open("r", encoding="utf-8") as fin, \
         dst.open("w", encoding="utf-8") as fout:

        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Ошибка JSON в строке {line_num}") from e

            question = obj.get("question", "").strip()
            answer = obj.get("answer", "").strip()

            if not question and not answer:
                continue

            body = ""
            if question:
                body += f"Вопрос: {question}\n"
            if answer:
                body += f"Ответ: {answer}"

            chunk = {
                "body": body
            }

            fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Готово: {dst}")

if __name__ == "__main__":
    convert_jsonl_to_chunks(INPUT_FILE, OUTPUT_FILE)