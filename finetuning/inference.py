import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

########################################
#              CONFIG                  #
########################################

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
LORA_ADAPTER = "phi3-mr-lora"  # Adapter path

SYSTEM_PROMPT = (
    "You are a navigation assistant helping the user locate a target object inside a building. "
    "You will receive a sequence of frames describing visible objects. " 
    "Each object includes: "
    "- the floor, "
    "- the relative position to the viewer, "
    "- the distance from the viewer, "
    "- and the room it belongs to. "
    
    "The frames appear in chronological order along the user's path from the starting point toward the target. "
    
    "Before starting the walk description, consider an initial turn direction if provided. "
    "Your task is to write a human-sounding description of the path. "
    "Avoid technical language or numeric measurements. Use intuitive guidance and stay under 120 words (using fewer words when possible). "
    
    "Mention at least one and at most two objects per room, choosing only the most informative for navigation. "
    "If the path includes stairs, simply write: 'go up/down the stairs to reach the <room_name>', without describing objects on the stairs. "
    
    "If you see the target location or object, mention it immediately and stop referencing any further objects. "
    
    "Only refer to objects that appear in the observations. Never invent or embellish details. "
    "Use the object IDs when referencing them (e.g., 'chair_5'). "
    
    "You will then receive a user question and the list of observations from the path, as well as the rooms visited in order. "
    "Imagine you are moving from the starting room to the target location, and provide clear path instructions."
)

SYSTEM_PROMPT = "You are a navigation assistant helping the user locate a target object inside a building.You will receive a sequence of frames describing visible objects. Each object includes:  - the floor,  - the relative position to the viewer,  - the distance from the viewer,  - and the room it belongs to.The frames appear in chronological order along the user's path from the starting point toward the target.Before starting the walk description, consider an initial turn direction if provided.Your task is to write a human-sounding description of the path.  Avoid technical language or numeric measurements. Use intuitive guidance and stay under 120 words (using fewer words when possible).Mention at least one and at most two objects per room, choosing only the most informative for navigation.  If the path includes stairs, simply write: \"go up/down the stairs to reach the <room_name>\", without describing objects on the stairs.If you see the target location or object, mention it immediately and stop referencing any further objects.Only refer to objects that appear in the observations. Never invent or embellish details.Use the object IDs when referencing them (e.g., 'chair_5').You will then receive a user question and the list of observations from the path, as well as the rooms visited in order. Imagine you are moving from the starting room to the target location, and provide clear path instructions."

USER_PROMPT = (
    "User question: where is the chair in the office?\n"
    "Observations:\n"
    "Initially, continue forward.\n"
    "In frame-000000 you are in upper bedroom on floor 1. You see wardrobe_26 [(relative position: lower-right), (room: upper bedroom)], door_1 [(relative position: center), (room: upper bedroom)].\n"
    "From previous frame, continue forward.\n"
    "In frame-000001 you are in office on floor 1. You see couch_125 [(relative position: lower-left), (room: office)], door_5 [(relative position: center-right), (room: office)].\n"
    "From previous frame, turn left.\n"
    "In frame-000002 you are in office on floor 1. You see chair_150 [(relative position: lower-left), (room: office)], chair_151 [(relative position: center), (room: office)].\n"
    "Rooms visited in order:\n"
    "upper bedroom (floor: 1), office (floor: 1)\n"
    "The user is in upper bedroom (floor: 1) and the target is in office (floor: 1)."
)

USER_PROMPT = "User question: Where is the wall clock?\n        Observations:\n        Initially, turn right.\n In frame-000000 you are in living room on floor 0. You see wall clock_205 [(relative position: center), (room: kitchen)], couch_126 [(relative position: lower-left), (room: living room)].\n        Rooms visited in order: \nliving room (floor: 0), kitchen (floor: 0)\n        The user is in living room (floor: 0) and the target is in kitchen (floor: 0)."

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
        max_new_tokens=150,
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
        temperature=0.2,
        top_p=0.85,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
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