import json
import re

def clean_text(text: str) -> str:
    """Remove newlines, excessive spaces, indentation."""
    # Rimuovi newline e tab
    text = text.replace("\n", " ").replace("\t", " ")

    # Collassa multipli spazi
    text = re.sub(r"\s+", " ", text)

    # Rimuovi spazi all'inizio e fine
    return text.strip()


def clean_jsonl(input_path: str, output_path: str):
    cleaned_samples = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                print("[WARNING] Skipping invalid JSON line:")
                print(line[:200])
                continue

            # Pulizia di tutti i messaggi
            for msg in sample["messages"]:
                msg["content"] = clean_text(msg["content"])

            cleaned_samples.append(sample)

    # Scrittura nuovo JSONL pulito
    with open(output_path, "w", encoding="utf-8") as f:
        for s in cleaned_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Cleaned {len(cleaned_samples)} samples → {output_path}")


if __name__ == "__main__":
    clean_jsonl("old_data.jsonl", "clean_old_data.jsonl")