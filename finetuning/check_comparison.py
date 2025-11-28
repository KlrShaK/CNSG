import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

########################################
#              CONFIG                  #
########################################

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
LORA_ADAPTER = "phi3-mr-lora-fixed"

########################################
#          LOAD MODELS                 #
########################################

print("=" * 80)
print(" MODEL PARAMETER VERIFICATION ".center(80))
print("=" * 80)

print("\n[1/3] Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)

print("[2/3] Loading finetuned model with LoRA adapter...")
finetuned_model = PeftModel.from_pretrained(
    base_model,
    LORA_ADAPTER,
    torch_dtype=torch.bfloat16,
)

print("[3/3] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

print("\n✅ All models loaded!\n")

########################################
#    CHECK 1: LoRA ADAPTER EXISTS      #
########################################

print("=" * 80)
print(" CHECK 1: LoRA Adapter Verification ".center(80))
print("=" * 80)

try:
    import os
    adapter_path = LORA_ADAPTER
    
    # Check for adapter files
    adapter_config = os.path.join(adapter_path, "adapter_config.json")
    adapter_weights = os.path.join(adapter_path, "adapter_model.safetensors")
    
    if os.path.exists(adapter_config):
        print(f"✅ Found adapter_config.json")
        with open(adapter_config, 'r') as f:
            import json
            config = json.load(f)
            print(f"   - LoRA rank (r): {config.get('r', 'N/A')}")
            print(f"   - LoRA alpha: {config.get('lora_alpha', 'N/A')}")
            print(f"   - Target modules: {config.get('target_modules', 'N/A')}")
    else:
        print(f"❌ adapter_config.json NOT found!")
    
    if os.path.exists(adapter_weights):
        file_size = os.path.getsize(adapter_weights) / (1024**2)  # MB
        print(f"✅ Found adapter_model.safetensors ({file_size:.2f} MB)")
    else:
        print(f"❌ adapter_model.safetensors NOT found!")
        
except Exception as e:
    print(f"❌ Error checking adapter files: {e}")

########################################
#    CHECK 2: PARAMETER COMPARISON     #
########################################

print("\n" + "=" * 80)
print(" CHECK 2: Parameter Modification Analysis ".center(80))
print("=" * 80)

print("\n🔍 Analyzing parameter differences...")

# Get all named parameters
base_params = dict(base_model.named_parameters())
finetuned_params = dict(finetuned_model.named_parameters())

print(f"\nBase model parameters: {len(base_params)}")
print(f"Finetuned model parameters: {len(finetuned_params)}")

# Find LoRA-specific parameters
lora_params = {name: param for name, param in finetuned_params.items() if 'lora' in name.lower()}

print(f"\n✅ Found {len(lora_params)} LoRA-specific parameters")

if len(lora_params) > 0:
    print("\n📋 LoRA Parameters:")
    total_lora_params = 0
    for name, param in list(lora_params.items())[:10]:  # Show first 10
        num_params = param.numel()
        total_lora_params += num_params
        print(f"   - {name}: {param.shape} ({num_params:,} params)")
    
    if len(lora_params) > 10:
        print(f"   ... and {len(lora_params) - 10} more LoRA parameters")
    
    # Count remaining params
    for name, param in list(lora_params.items())[10:]:
        total_lora_params += param.numel()
    
    print(f"\n📊 Total LoRA parameters: {total_lora_params:,}")
    print(f"   LoRA parameters size: {total_lora_params * 2 / (1024**2):.2f} MB (bfloat16)")
else:
    print("\n❌ WARNING: No LoRA parameters found! Fine-tuning may have failed!")

########################################
#    CHECK 3: WEIGHT STATISTICS        #
########################################

print("\n" + "=" * 80)
print(" CHECK 3: LoRA Weight Statistics ".center(80))
print("=" * 80)

if len(lora_params) > 0:
    print("\n📊 Analyzing LoRA weight distributions...\n")
    
    for name, param in list(lora_params.items())[:5]:  # Analyze first 5
        weights = param.detach().cpu().float().numpy().flatten()
        
        print(f"Parameter: {name}")
        print(f"  Shape: {param.shape}")
        print(f"  Mean: {weights.mean():.6f}")
        print(f"  Std:  {weights.std():.6f}")
        print(f"  Min:  {weights.min():.6f}")
        print(f"  Max:  {weights.max():.6f}")
        print(f"  Non-zero: {np.count_nonzero(weights):,} / {len(weights):,}")
        print()
    
    if len(lora_params) > 5:
        print(f"(Showing stats for first 5 LoRA parameters only)")
else:
    print("\n❌ No LoRA parameters to analyze!")

########################################
#    CHECK 4: INFERENCE COMPARISON     #
########################################

print("\n" + "=" * 80)
print(" CHECK 4: Inference Behavior Comparison ".center(80))
print("=" * 80)

SYSTEM_PROMPT = "You are a navigation assistant helping the user locate a target object inside a building."
TEST_PROMPT = "User question: How do I get to the kitchen?"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": TEST_PROMPT}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
input_len = inputs['input_ids'].shape[1]

print("\n🧪 Running inference test...")
print(f"Test prompt: '{TEST_PROMPT}'")

# Base model inference
print("\n⏳ Generating with BASE model...")
with torch.no_grad():
    base_output = base_model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,  # Deterministic
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,  # ✅ CAMBIATO: True invece di False
    )
base_response = tokenizer.decode(base_output[0][input_len:], skip_special_tokens=True)

# Finetuned model inference
print("⏳ Generating with FINETUNED model...")
with torch.no_grad():
    ft_output = finetuned_model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,  # Deterministic
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,  # ✅ CAMBIATO: True invece di False
    )
ft_response = tokenizer.decode(ft_output[0][input_len:], skip_special_tokens=True)

print("\n" + "─" * 80)
print("🔵 BASE MODEL:")
print("─" * 80)
print(base_response)

print("\n" + "─" * 80)
print("🟢 FINETUNED MODEL:")
print("─" * 80)
print(ft_response)

print("\n" + "─" * 80)
if base_response == ft_response:
    print("❌ WARNING: Responses are IDENTICAL!")
    print("   This suggests the LoRA adapter may not be properly loaded or trained!")
else:
    print("✅ Responses are DIFFERENT!")
    print("   This confirms the model has been modified by fine-tuning.")
print("─" * 80)

########################################
#    CHECK 5: LOGITS COMPARISON        #
########################################

print("\n" + "=" * 80)
print(" CHECK 5: Logits Distribution Analysis ".center(80))
print("=" * 80)

print("\n🔍 Comparing output logits for same input...")

# ✅ FIX: Usa return_dict=True e ottieni solo logits
with torch.no_grad():
    base_outputs = base_model(**inputs, use_cache=True, return_dict=True)
    base_logits = base_outputs.logits[0, -1, :]  # Last token logits
    
    ft_outputs = finetuned_model(**inputs, use_cache=True, return_dict=True)
    ft_logits = ft_outputs.logits[0, -1, :]

# Calculate difference
logits_diff = (ft_logits - base_logits).abs()
max_diff = logits_diff.max().item()
mean_diff = logits_diff.mean().item()

print(f"\nLogits difference statistics:")
print(f"  Max difference:  {max_diff:.6f}")
print(f"  Mean difference: {mean_diff:.6f}")
print(f"  Std difference:  {logits_diff.std().item():.6f}")

if max_diff < 1e-6:
    print("\n❌ WARNING: Logits are nearly identical!")
    print("   The model may not have been properly fine-tuned!")
else:
    print(f"\n✅ Logits are different (max diff: {max_diff:.6f})")
    print("   This confirms the model behavior has changed.")

# Top-k token comparison
k = 10
base_topk = base_logits.topk(k)
ft_topk = ft_logits.topk(k)

print(f"\n📊 Top-{k} most likely next tokens:")
print(f"\n{'BASE MODEL':<40} {'FINETUNED MODEL':<40}")
print("─" * 80)

for i in range(k):
    base_token = tokenizer.decode([base_topk.indices[i]])
    ft_token = tokenizer.decode([ft_topk.indices[i]])
    base_prob = torch.softmax(base_logits, dim=0)[base_topk.indices[i]].item()
    ft_prob = torch.softmax(ft_logits, dim=0)[ft_topk.indices[i]].item()
    
    marker = "✓" if base_token == ft_token else "✗"
    print(f"{i+1}. {base_token[:15]:<15} ({base_prob:.4f})    {marker}    {ft_token[:15]:<15} ({ft_prob:.4f})")

########################################
#          FINAL VERDICT               #
########################################

print("\n" + "=" * 80)
print(" FINAL VERDICT ".center(80))
print("=" * 80)

checks_passed = 0
total_checks = 5

# Check 1: LoRA files exist
import os
if os.path.exists(os.path.join(LORA_ADAPTER, "adapter_model.safetensors")):
    print("\n✅ CHECK 1 PASSED: LoRA adapter files exist")
    checks_passed += 1
else:
    print("\n❌ CHECK 1 FAILED: LoRA adapter files missing")

# Check 2: LoRA parameters found
if len(lora_params) > 0:
    print(f"✅ CHECK 2 PASSED: Found {len(lora_params)} LoRA parameters")
    checks_passed += 1
else:
    print("❌ CHECK 2 FAILED: No LoRA parameters found")

# Check 3: LoRA weights are non-zero
if len(lora_params) > 0:
    sample_param = list(lora_params.values())[0]
    if sample_param.abs().max().item() > 1e-6:
        print("✅ CHECK 3 PASSED: LoRA weights are non-zero")
        checks_passed += 1
    else:
        print("❌ CHECK 3 FAILED: LoRA weights are all near zero")
else:
    print("❌ CHECK 3 FAILED: No LoRA parameters to check")

# Check 4: Responses differ
if base_response != ft_response:
    print("✅ CHECK 4 PASSED: Model outputs differ")
    checks_passed += 1
else:
    print("❌ CHECK 4 FAILED: Model outputs are identical")

# Check 5: Logits differ
if max_diff > 1e-6:
    print(f"✅ CHECK 5 PASSED: Logits differ (max: {max_diff:.6f})")
    checks_passed += 1
else:
    print(f"❌ CHECK 5 FAILED: Logits are nearly identical")

print("\n" + "─" * 80)
print(f"TOTAL: {checks_passed}/{total_checks} checks passed")
print("─" * 80)

if checks_passed == total_checks:
    print("\n🎉 CONCLUSION: Fine-tuning was SUCCESSFUL!")
    print("   The model has been properly modified and shows different behavior.")
elif checks_passed >= 3:
    print("\n✅ CONCLUSION: Fine-tuning appears SUCCESSFUL with minor issues")
    print("   The model shows changes but some checks didn't pass completely.")
else:
    print("\n❌ CONCLUSION: Fine-tuning may have FAILED!")
    print("   The model shows minimal or no changes from the base model.")

print("\n" + "=" * 80)