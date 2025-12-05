import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

########################################
#              CONFIG                  #
########################################

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
LORA_ADAPTER = "phi3-mr-lora-fixed"  # Adapter path

SYSTEM_PROMPT = "You are a navigation assistant helping the user locate a target object inside a building.You will receive a sequence of frames describing visible objects. Each object includes:  - the floor,  - the relative position to the viewer,  - the distance from the viewer,  - and the room it belongs to.The frames appear in chronological order along the user's path from the starting point toward the target.Before starting the walk description, consider an initial turn direction if provided.Your task is to write a human-sounding description of the path.  Avoid technical language or numeric measurements. Use intuitive guidance and stay under 120 words (using fewer words when possible).Mention at least one and at most two objects per room, choosing only the most informative for navigation.  If the path includes stairs, simply write: “go up/down the stairs to reach the <room_name>”, without describing objects on the stairs.If you see the target location or object, mention it immediately and stop referencing any further objects.Only refer to objects that appear in the observations. Never invent or embellish details.Use the object IDs when referencing them (e.g., 'chair_5').You will then receive a user question and the list of observations from the path, as well as the rooms visited in order. Imagine you are moving from the starting room to the target location, and provide clear path instructions."
USER_PROMPT = "User question: How do I get to the upper bedroom? Observations: Initially, turn left. In frame-000000 you are in living room on floor 0. You see stairs_170 [(relative position: lower-center), (room: living room)], picture_136 [(relative position: center-left), (room: living room)]. From previous frame, continue forward. In frame-000001 you are in living room on floor 0. You see stairs_170 [(relative position: lower-center), (room: living room)], couch_125 [(relative position: lower-right), (room: office)]. From previous frame, continue forward. In frame-000002 you are in office on floor 1. You see door_1 [(relative position: center-right), (room: upper bedroom)], wardrobe_27 [(relative position: lower-right), (room: upper bedroom)]. From previous frame, continue forward. In frame-000003 you are in upper bedroom on floor 1. You see bed_40 [(relative position: center-left), (room: upper bedroom)], armchair_72 [(relative position: lower-right), (room: upper bedroom)]. Rooms visited in order: living room (floor: 0), office (floor: 1), upper bedroom (floor: 1) The user is in living room (floor: 0) and the target is in upper bedroom (floor: 1)."

########################################
#          LOAD MODEL                  #
########################################

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)

########################################
#    TEST MODELLO BASE (NO LORA)      #
########################################

print("\n" + "="*60)
print("TESTING BASE MODEL (without LoRA)")
print("="*60)

# Test con solo il base model
test_messages_simple = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_PROMPT}
]

prompt_base = tokenizer.apply_chat_template(
    test_messages_simple,
    tokenize=False,
    add_generation_prompt=True,
)

inputs_base = tokenizer(
    prompt_base,
    return_tensors="pt",
    truncation=True,
    max_length=2048,
).to(base_model.device)

input_len_base = inputs_base['input_ids'].shape[1]

print("Generating with BASE model...")
with torch.no_grad():
    outputs_base = base_model.generate(
        **inputs_base,
        max_new_tokens=120,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )

base_response = tokenizer.decode(
    outputs_base[0][input_len_base:], 
    skip_special_tokens=True
)

print("\n[BASE MODEL] Response:")
print(base_response)
print("="*60 + "\n")

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    LORA_ADAPTER,
    torch_dtype=torch.bfloat16,
)

model.eval()
print("Model loaded successfully!\n")

########################################
#          TEST SAMPLE                 #
########################################

test_messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    },
    {
        "role": "user", 
        "content": USER_PROMPT
    }
]

# Applica chat template
prompt = tokenizer.apply_chat_template(
    test_messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("=" * 60)
print("PROMPT (last 300 chars):")
print("=" * 60)
print("..." + prompt[-300:])
print("=" * 60)

# Tokenize
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=2048,
).to(model.device)

input_length = inputs['input_ids'].shape[1]
print(f"\nInput tokens: {input_length}")

########################################
#          GENERATE                    #
########################################

print("\nGenerating response...\n")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        #temperature=0.2,
        #top_p=0.85,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        #repetition_penalty=1.2,
        #no_repeat_ngram_size=3,
        use_cache=False,
    )

# ✅ FIX: Decodifica solo i token generati (escludendo l'input)
generated_tokens = outputs[0][input_length:]  # Prendi solo i nuovi token
generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print("=" * 60)
print("GENERATED ANSWER:")
print("=" * 60)
print(generated_text)
print("=" * 60)


###########

########################################
#         GENERAL INFERENCE FN         #
########################################

def run_inference(messages, title=""):
    print("\n" + "="*80)
    print(f" RUNNING SAMPLE: {title}")
    print("="*80)

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenization
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    input_length = inputs['input_ids'].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )

    generated = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    print("\n--- GENERATED RESPONSE ---")
    print(generated)
    print("--- END ---\n")

    return generated

########################################
#      FIVE TRAINING SAMPLES TEST      #
########################################

samples = [
    ("Sample 1", {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "User question: How do I get to the upper bedroom? Observations: Initially, turn left. In frame-000000 you are in living room on floor 0. You see stairs_170 [(relative position: lower-center), (room: living room)], picture_136 [(relative position: center-left), (room: living room)]. From previous frame, continue forward. In frame-000001 you are in living room on floor 0. You see stairs_170 [(relative position: lower-center), (room: living room)], couch_125 [(relative position: lower-right), (room: office)]. From previous frame, continue forward. In frame-000002 you are in office on floor 1. You see door_1 [(relative position: center-right), (room: upper bedroom)], wardrobe_27 [(relative position: lower-right), (room: upper bedroom)]. From previous frame, continue forward. In frame-000003 you are in upper bedroom on floor 1. You see bed_40 [(relative position: center-left), (room: upper bedroom)], armchair_72 [(relative position: lower-right), (room: upper bedroom)]. Rooms visited in order: living room (floor: 0), office (floor: 1), upper bedroom (floor: 1) The user is in living room (floor: 0) and the target is in upper bedroom (floor: 1)."}
        ]
    }),
    ("Sample 2", {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "User question: Where is the upper bathroom? Observations: Initially, turn left. In frame-000000 you are in office on floor 1. You see door_6 [(relative position: lower-left), (room: office)], door_7 [(relative position: upper-center), (room: office)]. From previous frame, turn left. In frame-000001, you see cabinet_33 [(relative position: lower-right), (room: upper bathroom)], door_3 [(relative position: lower-center), (room: upper bathroom)]. Rooms visited in order: office (floor: 1), upper bathroom (floor: 1) The user is in office (floor: 1) and the target is in upper bathroom (floor: 1)."}
        ]
    }),
    ("Sample 3", {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "User question: How do I reach the living room? Observations: Initially, turn right. In frame-000000 you are in office on floor 1. You see door_5 [(relative position: center-right), (room: office)], stairs_170 [(relative position: lower-center), (room: living room)]. From previous frame, continue forward. In frame-000001 you are in office on floor 1. You see stairs_170 [(relative position: lower-center), (room: living room)], armchair_73 [(relative position: lower-left), (room: living room)]. From previous frame, continue forward. In frame-000002 you are in living room on floor 0. You see stairs_170 [(relative position: lower-center), (room: living room)], door_8 [(relative position: center-left), (room: living room)]. From previous frame, continue forward. In frame-000003 you are in living room on floor 0. You see stairs_170 [(relative position: lower-center), (room: living room)], door_8 [(relative position: center-left), (room: living room)]. From previous frame, turn left. In frame-000004 you are in living room on floor 0. You see door_8 [(relative position: center-right), (room: living room)], armchair_73 [(relative position: center), (room: living room)]. From previous frame, continue forward. In frame-000005 you are in living room on floor 0. You see fireplace_182 [(relative position: lower-right), (room: living room)], armchair_73 [(relative position: lower-right), (room: living room)]. Rooms visited in order: office (floor: 1), living room (floor: 0) The user is in office (floor: 1) and the target is in living room (floor: 0)."}
        ]
    }),
    ("Sample 4", {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "User question: Where is the kitchen located? Observations: Initially, continue forward. In frame-000000 you are in office on floor 1. You see door_5 [(relative position: center-right), (room: office)], picture_135 [(relative position: center), (room: living room)]. From previous frame, continue forward. In frame-000001 you are in office on floor 1. You see stairs_170 [(relative position: lower-center), (room: living room)], door_8 [(relative position: lower-left), (room: living room)]. From previous frame, continue forward. In frame-000002 you are in living room on floor 0. You see stairs_170 [(relative position: lower-center), (room: living room)], door_8 [(relative position: center-left), (room: living room)]. From previous frame, continue forward. In frame-000003 you are in living room on floor 0. You see stairs_170 [(relative position: lower-center), (room: living room)], door_8 [(relative position: center-left), (room: living room)]. From previous frame, turn left. In frame-000004 you are in living room on floor 0. You see door_8 [(relative position: center-right), (room: living room)], armchair_73 [(relative position: center), (room: living room)]. From previous frame, turn left. In frame-000005 you are in living room on floor 0. You see couch_126 [(relative position: lower-left), (room: living room)], fireplace_182 [(relative position: lower-right), (room: living room)]. From previous frame, continue forward. In frame-000006 you are in living room on floor 0. You see armchair_74 [(relative position: lower-right), (room: living room)], couch_126 [(relative position: lower-left), (room: living room)]. From previous frame, continue forward. In frame-000007 you are in living room on floor 0. You see kitchen cabinet_208 [(relative position: center), (room: kitchen)], oven and stove_222 [(relative position: lower-right), (room: kitchen)]. Rooms visited in order: office (floor: 1), living room (floor: 0), kitchen (floor: 0) The user is in office (floor: 1) and the target is in kitchen (floor: 0)."}
        ]
    }),
    ("Sample 5", {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "User question: I need to find lower bathroom 1. Observations: Initially, turn right. In frame-000000 you are in entryway on floor 0. You see picture_140 [(relative position: center-left), (room: entryway)], door_15 [(relative position: center-right), (room: dining room)]. From previous frame, turn left. In frame-000001 you are in entryway on floor 0. You see door_12 [(relative position: center-right), (room: kitchen)], wall clock_205 [(relative position: center), (room: kitchen)]. From previous frame, continue forward. In frame-000002 you are in kitchen on floor 0. You see couch_126 [(relative position: lower-left), (room: living room)], door_11 [(relative position: center), (room: kitchen)]. From previous frame, turn right. In frame-000003 you are in kitchen on floor 0. You see door_11 [(relative position: lower-left), (room: kitchen)]. From previous frame, turn right. In frame-000004, you see door_11 [(relative position: lower-right), (room: kitchen)], bath sink_225 [(relative position: lower-left), (room: lower bathroom 1)]. Rooms visited in order: entryway (floor: 0), kitchen (floor: 0), lower bathroom 1 (floor: 0) The user is in entryway (floor: 0) and the target is in lower bathroom 1 (floor: 0)."}
        ]
    }),
]

########################################
#       RUN INFERENCE ON 5 SAMPLES     #
########################################

for title, sample in samples:
    run_inference(sample["messages"], title)