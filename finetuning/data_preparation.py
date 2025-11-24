import re
import json

INPUT_FILE = "raw_examples.txt"
OUTPUT_FILE = "finetune_data.jsonl"

SYSTEM_PROMPT = (
    "You are a navigation assistant helping the user locate a target object inside a building."
    "You will receive a sequence of frames describing visible objects. " 
    "Each object includes:  "
    "- the floor,  "
    "- the relative position to the viewer,  "
    "- the distance from the viewer,  "
    "- and the room it belongs to."

    "The frames appear in chronological order along the user's path from the starting point toward the target."

    "Before starting the walk description, consider an initial turn direction if provided."
    "Your task is to write a human-sounding description of the path.  "
    "Avoid technical language or numeric measurements. Use intuitive guidance and stay under 120 words (using fewer words when possible)."

    "Mention at least one and at most two objects per room, choosing only the most informative for navigation.  "
    "If the path includes stairs, simply write: “go up/down the stairs to reach the <room_name>”, without describing objects on the stairs."

    "If you see the target location or object, mention it immediately and stop referencing any further objects."

    "Only refer to objects that appear in the observations. Never invent or embellish details."
    "Use the object IDs when referencing them (e.g., 'chair_5')."

    "You will then receive a user question and the list of observations from the path, as well as the rooms visited in order. "
    "Imagine you are moving from the starting room to the target location, and provide clear path instructions."
)

def extract_examples(text: str):
    """Parse raw examples from the input text."""
    # Gli esempi iniziano con:   78.   oppure 79.   ecc.
    blocks = re.split(r"\n\s*\d+\.\s*\n", text)
    blocks = [b.strip() for b in blocks if b.strip()]

    parsed = []

    for block in blocks:
        # Estrarre User Prompt
        user_match = re.search(
            r"User Prompt:\s*(.*?)Generated Description:",
            block,
            flags=re.S | re.I,
        )
        if not user_match:
            print("[WARNING] Could not find user prompt in block:")
            print(block)
            continue

        user_prompt = user_match.group(1).strip()

        # Estrarre Output
        output_match = re.search(
            r"Generated Description:\s*(.*)$",
            block,
            flags=re.S | re.I,
        )
        if not output_match:
            print("[WARNING] Could not find response in block:")
            print(block)
            continue

        assistant_response = output_match.group(1).strip()

        # Costruire struttura dataset
        entry = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response},
            ]
        }

        parsed.append(entry)

    return parsed


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw = f.read()

    examples = extract_examples(raw)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[DONE] Parsed {len(examples)} examples into {OUTPUT_FILE}")


if __name__ == "__main__":
    main()