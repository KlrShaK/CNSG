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
DATA_PATH = "finetune_data.jsonl"  # <= dataset JSONL preparato da te
OUTPUT_DIR = "phi3-mr-lora"

# Training hyperparameters
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,       # eff. batch size 16
    learning_rate=1e-5,                  # molto meglio per dataset piccolo
    warmup_ratio=0.1,
    logging_steps=25,
    logging_strategy="steps",
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,                           # se GPU supporta, altrimenti fp16=True
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_grad_norm=1.0,
    weight_decay=0.01,
    report_to="none",
    seed=42,
)

# Standard LoRA for LLMs
peft_config = LoraConfig(
    r=16,
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

logger.info("===== Phi-3 LoRA Finetuning Started =====")
logger.info(f"Model: {MODEL_NAME}")
logger.info(f"Dataset: {DATA_PATH}")
logger.info(f"Output dir: {OUTPUT_DIR}")


########################################
#       LOAD MODEL & TOKENIZER          #
########################################

logger.info("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
    use_cache=False,
    trust_remote_code=True,
    attn_implementation="eager",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.model_max_length = 2048
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = "right"

logger.info("Model and tokenizer loaded.")


########################################
#               DATASET                #
########################################

logger.info("Loading dataset...")

raw_dataset = load_dataset(
    "json",
    data_files={"train": DATA_PATH}
)

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
        add_generation_prompt=False,
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


########################################
#               TRAINING               #
########################################

logger.info("Initializing SFTTrainer...")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    peft_config=peft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

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


########################################
#              SAVE MODEL              #
########################################

logger.info("Saving LoRA adapter...")
trainer.save_model(OUTPUT_DIR)

logger.info("===== Finetuning Completed =====")