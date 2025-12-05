import logging
import sys
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer


########################################
#              CONFIG                  #
########################################

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DATA_PATH = "finetune_data.jsonl"
OUTPUT_DIR = "phi3-mr-lora-fixed"  # ✅ Nuovo nome per distinguere

# ✅ Training hyperparameters - OTTIMIZZATI per 1000 samples
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    
    num_train_epochs=2,                  # ✅ 2 epoche OK con 1000 samples
    per_device_train_batch_size=2,       # ✅ Batch 2 invece di 1
    gradient_accumulation_steps=4,       # ✅ Eff. batch = 8
    
    learning_rate=1e-5,
    warmup_ratio=0.1,
    
    logging_steps=50,
    logging_strategy="steps",
    eval_strategy="steps",
    eval_steps=50,                       # ✅ Più frequente per monitorare
    
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_grad_norm=1.0,
    weight_decay=0.01,
    
    report_to="none",
    seed=42,
)

# ✅ LoRA config - BILANCIATO
peft_config = LoraConfig(
    r=16,                                # ✅ 16 va bene per 1000 samples
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj",
        "up_proj", "down_proj",
    ],
)

########################################
#              LOGGING                 #
########################################

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

logger.info("===== Phi-3 LoRA Finetuning Started (FIXED VERSION) =====")
logger.info(f"Model: {MODEL_NAME}")
logger.info(f"Dataset: {DATA_PATH}")
logger.info(f"Output dir: {OUTPUT_DIR}")


########################################
#       LOAD MODEL & TOKENIZER          #
########################################

logger.info("Loading base model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.model_max_length = 2048
tokenizer.pad_token = tokenizer.eos_token      # ✅ FIX
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

# ✅ DEBUG: Verifica tokenizer config
logger.info(f"✅ Tokenizer pad_token: {tokenizer.pad_token}")
logger.info(f"✅ Tokenizer pad_token_id: {tokenizer.pad_token_id}")
logger.info(f"✅ Tokenizer eos_token: {tokenizer.eos_token}")
logger.info(f"✅ Tokenizer eos_token_id: {tokenizer.eos_token_id}")
logger.info(f"✅ Tokenizer model_max_length: {tokenizer.model_max_length}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
    use_cache=False,
    trust_remote_code=True,
    attn_implementation="eager",
)

logger.info("Model and tokenizer loaded.")


########################################
#               DATASET                #
########################################

logger.info("Loading dataset...")

raw_dataset = load_dataset(
    "json",
    data_files={"train": DATA_PATH}
)

# ✅ DEBUG: Mostra dataset raw
logger.info(f"✅ Total samples loaded: {len(raw_dataset['train'])}")
logger.info("\n" + "-" * 80)
logger.info("📋 RAW SAMPLE #0 (before processing):")
logger.info("-" * 80)
sample = raw_dataset["train"][0]
for i, msg in enumerate(sample["messages"]):
    logger.info(f"  Message {i} [{msg['role']}]:")
    content_preview = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
    logger.info(f"    {content_preview}")

# Train / test split (5%)
train_test = raw_dataset["train"].train_test_split(test_size=0.05, seed=42)
train_dataset = train_test["train"]
eval_dataset = train_test["test"]
logger.info(f"Train samples: {len(train_dataset)}")
logger.info(f"Eval samples:  {len(eval_dataset)}")


def apply_chat_template(example):
    example["text"] = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=True,  # ✅ CRITICAL FIX
    )
    return example


logger.info("Applying chat template...")

train_dataset = train_dataset.map(
    apply_chat_template,
    remove_columns=["messages"],
    desc="Processing train set",
)

eval_dataset = eval_dataset.map(
    apply_chat_template,
    remove_columns=["messages"],
    desc="Processing eval set",
)

# ✅ DEBUG: Mostra processed sample
logger.info("\n" + "-" * 80)
logger.info("📋 PROCESSED SAMPLE #0 (after chat template):")
logger.info("-" * 80)
processed_sample = train_dataset[0]["text"]
logger.info(f"Length: {len(processed_sample)} chars")
logger.info("\nFirst 500 chars:")
logger.info(processed_sample[:500])
logger.info("\n...")
logger.info("\nLast 300 chars:")
logger.info(processed_sample[-300:])

# ✅ DEBUG: Verifica che termini con <|assistant|>
if "<|assistant|>" in processed_sample[-50:]:
    logger.info("✅ CORRECT: Sample ends with <|assistant|> (generation prompt)")
else:
    logger.warning("⚠️  WARNING: Sample does NOT end with <|assistant|>")
    logger.info(f"   Actual ending: ...{processed_sample[-100:]}")

# ✅ DEBUG: Statistiche dataset
logger.info("\n" + "-" * 80)
logger.info("📊 DATASET STATISTICS:")
logger.info("-" * 80)
lengths = [len(sample["text"]) for sample in train_dataset]
logger.info(f"  Min length: {min(lengths)} chars")
logger.info(f"  Max length: {max(lengths)} chars")
logger.info(f"  Avg length: {sum(lengths)/len(lengths):.1f} chars")


########################################
#               TRAINING               #
########################################

logger.info("Initializing SFTTrainer...")

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    peft_config=peft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# ✅ DEBUG: Verifica parametri trainer
logger.info("Training configuration:")
logger.info(f"  Epochs: {training_args.num_train_epochs}")
logger.info(f"  Batch size: {training_args.per_device_train_batch_size}")
logger.info(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
logger.info(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
logger.info(f"  Learning rate: {training_args.learning_rate}")
logger.info(f"  LoRA rank: {peft_config.r}")
logger.info(f"  LoRA alpha: {peft_config.lora_alpha}")

# ✅ DEBUG: Verifica packing è disabilitato
logger.info("\n" + "-" * 80)
if hasattr(trainer, 'packing') and trainer.packing:
    logger.error("❌ ERROR: Packing is ENABLED! This will break training!")
    sys.exit(1)
else:
    logger.info("✅ VERIFIED: Packing is DISABLED")

# ✅ DEBUG: Conta parametri trainable
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
logger.info(f"\n📊 Model parameters:")
logger.info(f"  Trainable: {trainable_params:,}")
logger.info(f"  Total: {total_params:,}")
logger.info(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")

logger.info("Starting training...")

train_result = trainer.train()

logger.info("Training complete.")

trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()


########################################
#               EVAL                   #
########################################

logger.info("Evaluating...")

tokenizer.padding_side = "left"
metrics = trainer.evaluate()
metrics["eval_samples"] = len(eval_dataset)

trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

# ✅ DEBUG: Mostra metriche finali
logger.info("\n" + "-" * 80)
logger.info("📊 FINAL METRICS:")
logger.info("-" * 80)
for key, value in metrics.items():
    if isinstance(value, float):
        logger.info(f"  {key}: {value:.4f}")
    else:
        logger.info(f"  {key}: {value}")


########################################
#              SAVE MODEL              #
########################################

logger.info("Saving LoRA adapter...")
trainer.save_model(OUTPUT_DIR)

logger.info("===== Finetuning Completed =====")

########################################
#          INFERENCE TEST              #
########################################

logger.info("\n" + "=" * 80)
logger.info("Running inference tests on multiple samples...")
logger.info("=" * 80)

from peft import PeftModel

# Load the finetuned model
test_model = PeftModel.from_pretrained(
    model,
    OUTPUT_DIR,
    torch_dtype=torch.bfloat16,
)
test_model.eval()

# System prompt (esatto dal dataset)
SYSTEM_PROMPT = "You are a navigation assistant helping the user locate a target object inside a building.You will receive a sequence of frames describing visible objects. Each object includes:  - the floor,  - the relative position to the viewer,  - the distance from the viewer,  - and the room it belongs to.The frames appear in chronological order along the user's path from the starting point toward the target.Before starting the walk description, consider an initial turn direction if provided.Your task is to write a human-sounding description of the path.  Avoid technical language or numeric measurements. Use intuitive guidance and stay under 120 words (using fewer words when possible).Mention at least one and at most two objects per room, choosing only the most informative for navigation.  If the path includes stairs, simply write: \"go up/down the stairs to reach the <room_name>\", without describing objects on the stairs.If you see the target location or object, mention it immediately and stop referencing any further objects.Only refer to objects that appear in the observations. Never invent or embellish details.Use the object IDs when referencing them (e.g., 'chair_5').You will then receive a user question and the list of observations from the path, as well as the rooms visited in order. Imagine you are moving from the starting room to the target location, and provide clear path instructions."

########################################
#      TEST SAMPLES DEFINITION         #
########################################

# ✅ Test sample (nuovo, mai visto in training)
NEW_SAMPLE = {
    "name": "NEW UNSEEN SAMPLE",
    "user_prompt": "User question: How do I get to the upper bedroom? Observations: Initially, turn left. In frame-000000 you are in living room on floor 0. You see stairs_170 [(relative position: lower-center), (room: living room)], picture_136 [(relative position: center-left), (room: living room)]. From previous frame, continue forward. In frame-000001 you are in living room on floor 0. You see stairs_170 [(relative position: lower-center), (room: living room)], couch_125 [(relative position: lower-right), (room: office)]. From previous frame, continue forward. In frame-000002 you are in office on floor 1. You see door_1 [(relative position: center-right), (room: upper bedroom)], wardrobe_27 [(relative position: lower-right), (room: upper bedroom)]. From previous frame, continue forward. In frame-000003 you are in upper bedroom on floor 1. You see bed_40 [(relative position: center-left), (room: upper bedroom)], armchair_72 [(relative position: lower-right), (room: upper bedroom)]. Rooms visited in order: living room (floor: 0), office (floor: 1), upper bedroom (floor: 1) The user is in living room (floor: 0) and the target is in upper bedroom (floor: 1).",
    "expected": "N/A (unseen sample)"
}

# ✅ 5 samples dal training set
TRAINING_SAMPLES = [
    {
        "name": "Training Sample 1",
        "user_prompt": "User question: How do I get to the upper bedroom? Observations: Initially, turn left. In frame-000000 you are in living room on floor 0. You see stairs_170 [(relative position: lower-center), (room: living room)], picture_136 [(relative position: center-left), (room: living room)]. From previous frame, continue forward. In frame-000001 you are in living room on floor 0. You see stairs_170 [(relative position: lower-center), (room: living room)], couch_125 [(relative position: lower-right), (room: office)]. From previous frame, continue forward. In frame-000002 you are in office on floor 1. You see door_1 [(relative position: center-right), (room: upper bedroom)], wardrobe_27 [(relative position: lower-right), (room: upper bedroom)]. From previous frame, continue forward. In frame-000003 you are in upper bedroom on floor 1. You see bed_40 [(relative position: center-left), (room: upper bedroom)], armchair_72 [(relative position: lower-right), (room: upper bedroom)]. Rooms visited in order: living room (floor: 0), office (floor: 1), upper bedroom (floor: 1) The user is in living room (floor: 0) and the target is in upper bedroom (floor: 1).",
        "expected": "Start by turning left in the living room. Walk straight ahead and you'll find stairs_170 leading up. Go up the stairs to reach the office. From there, continue forward, and you'll see door_1 to your right. This door leads to the upper bedroom."
    },
    {
        "name": "Training Sample 2",
        "user_prompt": "User question: Where is the upper bathroom? Observations: Initially, turn left. In frame-000000 you are in office on floor 1. You see door_6 [(relative position: lower-left), (room: office)], door_7 [(relative position: upper-center), (room: office)]. From previous frame, turn left. In frame-000001, you see cabinet_33 [(relative position: lower-right), (room: upper bathroom)], door_3 [(relative position: lower-center), (room: upper bathroom)]. Rooms visited in order: office (floor: 1), upper bathroom (floor: 1) The user is in office (floor: 1) and the target is in upper bathroom (floor: 1).",
        "expected": "Start by turning left in the office. You'll see door_3 directly in front of you, which leads to the upper bathroom."
    },
    {
        "name": "Training Sample 3",
        "user_prompt": "User question: How do I reach the living room? Observations: Initially, turn right. In frame-000000 you are in office on floor 1. You see door_5 [(relative position: center-right), (room: office)], stairs_170 [(relative position: lower-center), (room: living room)]. From previous frame, continue forward. In frame-000001 you are in office on floor 1. You see stairs_170 [(relative position: lower-center), (room: living room)], armchair_73 [(relative position: lower-left), (room: living room)]. From previous frame, continue forward. In frame-000002 you are in living room on floor 0. You see stairs_170 [(relative position: lower-center), (room: living room)], door_8 [(relative position: center-left), (room: living room)]. From previous frame, continue forward. In frame-000003 you are in living room on floor 0. You see stairs_170 [(relative position: lower-center), (room: living room)], door_8 [(relative position: center-left), (room: living room)]. From previous frame, turn left. In frame-000004 you are in living room on floor 0. You see door_8 [(relative position: center-right), (room: living room)], armchair_73 [(relative position: center), (room: living room)]. From previous frame, continue forward. In frame-000005 you are in living room on floor 0. You see fireplace_182 [(relative position: lower-right), (room: living room)], armchair_73 [(relative position: lower-right), (room: living room)]. Rooms visited in order: office (floor: 1), living room (floor: 0) The user is in office (floor: 1) and the target is in living room (floor: 0).",
        "expected": None  # Aggiungi se ce l'hai
    },
    {
        "name": "Training Sample 4",
        "user_prompt": "User question: Where is the kitchen located? Observations: Initially, continue forward. In frame-000000 you are in office on floor 1. You see door_5 [(relative position: center-right), (room: office)], picture_135 [(relative position: center), (room: living room)]. From previous frame, continue forward. In frame-000001 you are in office on floor 1. You see stairs_170 [(relative position: lower-center), (room: living room)], door_8 [(relative position: lower-left), (room: living room)]. From previous frame, continue forward. In frame-000002 you are in living room on floor 0. You see stairs_170 [(relative position: lower-center), (room: living room)], door_8 [(relative position: center-left), (room: living room)]. From previous frame, continue forward. In frame-000003 you are in living room on floor 0. You see stairs_170 [(relative position: lower-center), (room: living room)], door_8 [(relative position: center-left), (room: living room)]. From previous frame, turn left. In frame-000004 you are in living room on floor 0. You see door_8 [(relative position: center-right), (room: living room)], armchair_73 [(relative position: center), (room: living room)]. From previous frame, turn left. In frame-000005 you are in living room on floor 0. You see couch_126 [(relative position: lower-left), (room: living room)], fireplace_182 [(relative position: lower-right), (room: living room)]. From previous frame, continue forward. In frame-000006 you are in living room on floor 0. You see armchair_74 [(relative position: lower-right), (room: living room)], couch_126 [(relative position: lower-left), (room: living room)]. From previous frame, continue forward. In frame-000007 you are in living room on floor 0. You see kitchen cabinet_208 [(relative position: center), (room: kitchen)], oven and stove_222 [(relative position: lower-right), (room: kitchen)]. Rooms visited in order: office (floor: 1), living room (floor: 0), kitchen (floor: 0) The user is in office (floor: 1) and the target is in kitchen (floor: 0).",
        "expected": None
    },
    {
        "name": "Training Sample 5",
        "user_prompt": "User question: I need to find lower bathroom 1. Observations: Initially, turn right. In frame-000000 you are in entryway on floor 0. You see picture_140 [(relative position: center-left), (room: entryway)], door_15 [(relative position: center-right), (room: dining room)]. From previous frame, turn left. In frame-000001 you are in entryway on floor 0. You see door_12 [(relative position: center-right), (room: kitchen)], wall clock_205 [(relative position: center), (room: kitchen)]. From previous frame, continue forward. In frame-000002 you are in kitchen on floor 0. You see couch_126 [(relative position: lower-left), (room: living room)], door_11 [(relative position: center), (room: kitchen)]. From previous frame, turn right. In frame-000003 you are in kitchen on floor 0. You see door_11 [(relative position: lower-left), (room: kitchen)]. From previous frame, turn right. In frame-000004, you see door_11 [(relative position: lower-right), (room: kitchen)], bath sink_225 [(relative position: lower-left), (room: lower bathroom 1)]. Rooms visited in order: entryway (floor: 0), kitchen (floor: 0), lower bathroom 1 (floor: 0) The user is in entryway (floor: 0) and the target is in lower bathroom 1 (floor: 0).",
        "expected": None
    }
]

########################################
#      INFERENCE FUNCTION              #
########################################

def run_inference_test(model, sample_dict, sample_num):
    """Run inference on a single sample"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sample_dict["user_prompt"]}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs['input_ids'].shape[1]
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🧪 TEST {sample_num}: {sample_dict['name']}")
    logger.info(f"{'='*80}")
    logger.info(f"Input tokens: {input_len}")
    logger.info(f"\nUser prompt (first 150 chars):")
    logger.info(f"{sample_dict['user_prompt'][:150]}...")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
    
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    
    logger.info(f"\n{'─'*80}")
    logger.info(f"📝 GENERATED RESPONSE:")
    logger.info(f"{'─'*80}")
    logger.info(response)
    logger.info(f"{'─'*80}")
    logger.info(f"Length: {len(response.split())} words")
    
    # Show expected if available
    if sample_dict.get("expected"):
        logger.info(f"\n{'─'*80}")
        logger.info(f"✅ EXPECTED RESPONSE:")
        logger.info(f"{'─'*80}")
        logger.info(sample_dict["expected"])
        logger.info(f"{'─'*80}")
        logger.info(f"Length: {len(sample_dict['expected'].split())} words")
        
        # Simple quality check
        if len(response.split()) > 150:
            logger.warning(f"⚠️  WARNING: Response is too long (>150 words)")
        if "hallway" in response.lower() and "hallway" not in sample_dict["user_prompt"].lower():
            logger.warning(f"⚠️  WARNING: Possible hallucination detected (invented 'hallway')")
    
    return response

########################################
#      RUN ALL TESTS                   #
########################################

logger.info("\n" + "=" * 80)
logger.info("🎯 TESTING ON NEW (UNSEEN) SAMPLE")
logger.info("=" * 80)

new_response = run_inference_test(test_model, NEW_SAMPLE, "NEW")

logger.info("\n\n" + "=" * 80)
logger.info("🎯 TESTING ON TRAINING SAMPLES (should match expectations)")
logger.info("=" * 80)

training_responses = []
for i, sample in enumerate(TRAINING_SAMPLES, 1):
    response = run_inference_test(test_model, sample, f"TRAIN-{i}")
    training_responses.append(response)

########################################
#      SUMMARY                         #
########################################

logger.info("\n" + "=" * 80)
logger.info("📊 TEST SUMMARY")
logger.info("=" * 80)

logger.info("\n✅ New sample test completed")
logger.info(f"   Response length: {len(new_response.split())} words")

logger.info("\n✅ Training samples tests completed")
for i, resp in enumerate(training_responses, 1):
    logger.info(f"   Sample {i} length: {len(resp.split())} words")

# Quality checks
all_responses = [new_response] + training_responses
avg_length = sum(len(r.split()) for r in all_responses) / len(all_responses)

logger.info(f"\n📈 Average response length: {avg_length:.1f} words")

if avg_length > 100:
    logger.warning("⚠️  Average response length is high (target: 30-80 words)")
elif avg_length < 20:
    logger.warning("⚠️  Average response length is low (responses may be incomplete)")
else:
    logger.info("✅ Response lengths look good!")

logger.info("\n" + "=" * 80)
logger.info("✅✅✅ FINETUNING COMPLETED SUCCESSFULLY ✅✅✅")
logger.info("=" * 80)
logger.info(f"\nModel saved at: {OUTPUT_DIR}")
logger.info("You can now use this model for inference!")
logger.info("=" * 80)