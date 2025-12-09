import torch
import json
import random
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

########################################
#              CONFIG                  #
########################################

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
LORA_ADAPTER = "phi3-mr-lora-fixed-v3"
TRAINING_DATA = "finetune_data.jsonl"
NUM_SAMPLES = 15

########################################
#          LOAD DATA                   #
########################################

print("Loading training data...")
with open(TRAINING_DATA, 'r') as f:
    all_samples = [json.loads(line) for line in f]

# Random sample
random.seed(42)
test_samples = random.sample(all_samples, NUM_SAMPLES)
print(f"Selected {NUM_SAMPLES} random samples from {len(all_samples)} total\n")

########################################
#          LOAD MODELS                 #
########################################

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)

print("Loading LoRA finetuned model...")
finetuned_model = PeftModel.from_pretrained(
    base_model,
    LORA_ADAPTER,
    torch_dtype=torch.bfloat16,
)
finetuned_model.eval()

print("✅ Models loaded!\n")

########################################
#       INFERENCE FUNCTION             #
########################################

def generate_response(model, messages, tokenizer):
    """Generate response from a model"""
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)
    
    input_len = inputs['input_ids'].shape[1]
    
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
    return response

########################################
#          RUN COMPARISON              #
########################################

print("=" * 120)
print(" " * 45 + "BASE vs FINETUNED COMPARISON")
print("=" * 120)

results = []

for idx, sample in enumerate(test_samples, 1):
    messages = sample["messages"]
    
    # Extract system, user, and expected answer
    system_content = messages[0]["content"]  # System prompt
    user_content = messages[1]["content"]     # User message
    expected = messages[2]["content"] if len(messages) > 2 else "N/A"  # Assistant message
    
    print(f"\n{'═' * 120}")
    print(f"{'═' * 120}")
    print(f" SAMPLE {idx}/{NUM_SAMPLES}".center(120))
    print(f"{'═' * 120}")
    print(f"{'═' * 120}")
    
    # Show FULL system prompt
    print(f"\n{'─' * 120}")
    print(f"🔧 SYSTEM PROMPT:")
    print(f"{'─' * 120}")
    print(system_content)
    
    # Show FULL user query
    print(f"\n{'─' * 120}")
    print(f"❓ USER QUERY:")
    print(f"{'─' * 120}")
    print(user_content)
    
    print(f"\n{'─' * 120}")
    
    # Generate with BASE model
    print("⏳ Generating with BASE model...")
    base_response = generate_response(base_model, messages[:2], tokenizer)
    
    # Generate with FINETUNED model
    print("⏳ Generating with FINETUNED model...")
    finetuned_response = generate_response(finetuned_model, messages[:2], tokenizer)
    
    # Display results
    print(f"\n{'━' * 120}")
    print(f"🔵 BASE MODEL OUTPUT ({len(base_response.split())} words):")
    print(f"{'━' * 120}")
    print(base_response)
    
    print(f"\n{'━' * 120}")
    print(f"🟢 FINETUNED MODEL OUTPUT ({len(finetuned_response.split())} words):")
    print(f"{'━' * 120}")
    print(finetuned_response)
    
    print(f"\n{'━' * 120}")
    print(f"✅ EXPECTED GROUND TRUTH ({len(expected.split())} words):")
    print(f"{'━' * 120}")
    print(expected)
    print(f"{'━' * 120}")
    
    # Store results
    results.append({
        'base_len': len(base_response.split()),
        'finetuned_len': len(finetuned_response.split()),
        'expected_len': len(expected.split()),
        'base_response': base_response,
        'finetuned_response': finetuned_response,
        'expected': expected,
        'user_query': user_content
    })

########################################
#          SUMMARY STATS               #
########################################

print("\n" + "=" * 120)
print("=" * 120)
print(" 📊 SUMMARY STATISTICS ".center(120))
print("=" * 120)
print("=" * 120)

avg_base = sum(r['base_len'] for r in results) / len(results)
avg_finetuned = sum(r['finetuned_len'] for r in results) / len(results)
avg_expected = sum(r['expected_len'] for r in results) / len(results)

print(f"\n📏 AVERAGE RESPONSE LENGTH:")
print(f"   Base Model:      {avg_base:>6.1f} words")
print(f"   Finetuned Model: {avg_finetuned:>6.1f} words")
print(f"   Expected:        {avg_expected:>6.1f} words")

min_base = min(r['base_len'] for r in results)
max_base = max(r['base_len'] for r in results)
min_ft = min(r['finetuned_len'] for r in results)
max_ft = max(r['finetuned_len'] for r in results)
min_exp = min(r['expected_len'] for r in results)
max_exp = max(r['expected_len'] for r in results)

print(f"\n📊 LENGTH RANGE:")
print(f"   Base Model:      min={min_base:>3}, max={max_base:>3}")
print(f"   Finetuned Model: min={min_ft:>3}, max={max_ft:>3}")
print(f"   Expected:        min={min_exp:>3}, max={max_exp:>3}")

# Quality checks
over_120_base = sum(1 for r in results if r['base_len'] > 120)
over_120_ft = sum(1 for r in results if r['finetuned_len'] > 120)

print(f"\n⚠️  RESPONSES EXCEEDING 120 WORDS:")
print(f"   Base Model:      {over_120_base}/{len(results)}")
print(f"   Finetuned Model: {over_120_ft}/{len(results)}")

if over_120_ft == 0:
    print(f"\n✅ All finetuned responses are within 120 word limit!")
else:
    print(f"\n⚠️  {over_120_ft} finetuned response(s) exceeded the limit")

# Length difference analysis
print(f"\n📉 LENGTH IMPROVEMENT:")
avg_diff = avg_expected - avg_finetuned
print(f"   Finetuned vs Expected: {avg_diff:+.1f} words difference")
if abs(avg_diff) < 10:
    print(f"   ✅ Finetuned model matches expected length well!")
elif avg_diff > 0:
    print(f"   ℹ️  Finetuned responses are slightly shorter than expected")
else:
    print(f"   ℹ️  Finetuned responses are slightly longer than expected")

print("\n" + "=" * 120)
print(" ✅ COMPARISON COMPLETE ".center(120))
print("=" * 120)