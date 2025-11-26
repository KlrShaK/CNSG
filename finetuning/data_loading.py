import re
import json

INPUT_FILE = "../habitat-sim/data_collection.jsonl"
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

# Regex per estrarre qualunque oggetto JSON {...}
JSON_OBJ = re.compile(r'\{.*?\}', re.S)

def convert(input_path: str, output_path: str):
    samples = []
    buffer = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # trova TUTTI gli oggetti JSON sulla riga
            matches = JSON_OBJ.findall(line)

            for m in matches:
                try:
                    obj = json.loads(m)
                except json.JSONDecodeError:
                    print("[WARNING] Skipping invalid JSON:", m[:120])
                    continue

                # costruiamo coppie user → assistant
                if obj["role"] == "user":
                    buffer = [obj]

                elif obj["role"] == "assistant":
                    if buffer and buffer[0]["role"] == "user":
                        user_msg = buffer[0]
                        assistant_msg = obj

                        entry = {
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_msg["content"]},
                                {"role": "assistant", "content": assistant_msg["content"]}
                            ]
                        }
                        samples.append(entry)
                        buffer = []  # reset

    # Scrivi il JSONL finale
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Converted {len(samples)} samples to {output_path}")


def main():
    convert(INPUT_FILE, OUTPUT_FILE)


if __name__ == "__main__":
    main()