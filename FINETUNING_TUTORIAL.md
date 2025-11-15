# Fine-Tuning GPT-OSS-120B on H200 SXM 2x GPU: Complete Tutorial

This tutorial provides a step-by-step guide to fine-tune the GPT-OSS-120B model using LoRA (Low-Rank Adaptation) on 2x NVIDIA H200 SXM GPUs.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Hardware Requirements](#hardware-requirements)
3. [Environment Setup](#environment-setup)
4. [Session Storage Configuration](#session-storage-configuration)
5. [Data Preparation](#data-preparation)
6. [Model Loading](#model-loading)
7. [LoRA Configuration](#lora-configuration)
8. [Training Setup](#training-setup)
9. [Running Training](#running-training)
10. [Saving Checkpoints](#saving-checkpoints)
11. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Accounts
- **HuggingFace Account**: You need a HuggingFace account with access to the GPT-OSS-120B model
- **HuggingFace Token**: Generate an access token from [HuggingFace Settings](https://huggingface.co/settings/tokens)

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.8 or higher
- **CUDA**: 11.8 or higher
- **NVIDIA Drivers**: Latest compatible drivers for H200 GPUs

---

## Hardware Requirements

### Minimum Configuration
- **GPUs**: 2x NVIDIA H200 SXM (141GB each)
- **Total GPU Memory**: 282GB
- **System RAM**: 128GB+ recommended
- **Storage**: 500GB+ free space (for model weights, cache, and checkpoints)
- **Network**: Stable internet connection for model download (~240GB)

### Why 2x H200?
The GPT-OSS-120B model requires approximately 240GB of memory in 4-bit quantized format. With 2x H200 GPUs (282GB total), we can:
- Load the model across both GPUs
- Maintain 6GB headroom per GPU for activations and gradients
- Enable efficient distributed training with DeepSpeed

---

## Environment Setup

### Step 1: Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv finetuning_env
source finetuning_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 2: Install PyTorch with CUDA Support

```bash
# Install PyTorch (adjust CUDA version if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Install Required Libraries

```bash
# Core ML libraries
pip install -U "transformers>=4.35.0"
pip install accelerate
pip install deepspeed
pip install bitsandbytes
pip install peft
pip install datasets
pip install sentence-transformers
pip install safetensors

# Optional but recommended
pip install hf_transfer  # Faster model downloads
pip install flash-attn  # Flash Attention 2 (optional, requires compilation)
```

### Step 4: Verify Installation

```bash
# Check GPU availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Verify libraries
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"
python3 -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
python3 -c "import peft; print(f'PEFT: {peft.__version__}')"
```

### Step 5: Set HuggingFace Token

```bash
# Method 1: Environment variable (recommended)
export HF_TOKEN="your_huggingface_token_here"
export HUGGINGFACE_TOKEN="your_huggingface_token_here"

# Method 2: Login via CLI
huggingface-cli login
```

---

## Session Storage Configuration

### Understanding Storage Requirements

The GPT-OSS-120B model requires significant storage:
- **Model weights (quantized)**: ~240GB
- **HuggingFace cache**: ~250GB (includes tokenizers, configs)
- **Training checkpoints**: ~1-5GB per checkpoint
- **Training data**: Varies by dataset size

### Configure HuggingFace Cache Directory

By default, HuggingFace caches models in `~/.cache/huggingface/`. For large models, you may want to use a different location with more space:

```python
import os

# Set custom cache directory (adjust path as needed)
custom_cache = "/path/to/large/storage/hf_cache"
os.makedirs(custom_cache, exist_ok=True)

# Set environment variables
os.environ["HF_HOME"] = custom_cache
os.environ["TRANSFORMERS_CACHE"] = os.path.join(custom_cache, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(custom_cache, "datasets")
```

### Check Available Disk Space

```bash
# Check disk space
df -h

# Check cache directory size
du -sh ~/.cache/huggingface/
```

### Clean Old Cache (if needed)

```bash
# Remove old model caches (be careful!)
# Only remove if you're sure you don't need them
rm -rf ~/.cache/huggingface/hub/models--*  # Remove all cached models
# Or remove specific model:
# rm -rf ~/.cache/huggingface/hub/models--openai--gpt-oss-120b
```

---

## Data Preparation

### Data Format

Your training data should be in JSON format. Each JSON file should contain a list of training examples. Each example should have the following structure:

```json
{
  "score": 11,
  "max_score": 30,
  "normalized_score": 0.37,
  "rubric_alignment": [
    {
      "question": "Your question text here",
      "answer": "Student answer text here",
      "score": 3,
      "max_score": 5
    }
  ],
  "feedback": "Optional feedback text"
}
```

### Organize Data Files

```bash
# Create data directory
mkdir -p data

# Place your JSON files in the data directory
# Example: data/train_1.json, data/train_2.json, etc.
```

### Data Loading Code

```python
from datasets import load_dataset

# Load all JSON files from data directory
data_files = {"train": "./data/*.json"}
ds = load_dataset("json", data_files=data_files)
print(f"Loaded {len(ds['train'])} training examples")
```

---

## Model Loading

### Step 1: Configure GPU Memory

```python
import torch
import os

# Verify 2 GPUs are available
num_gpus = torch.cuda.device_count()
assert num_gpus >= 2, f"Expected 2 GPUs, found {num_gpus}"

# Print GPU information
for i in range(num_gpus):
    prop = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {prop.name} - {round(prop.total_memory/1024**3, 2)} GiB")

# Configure memory allocation (135GB per GPU, leaving 6GB headroom)
max_memory = {}
for i in range(2):
    max_memory[i] = "135GiB"
max_memory["cpu"] = "50GiB"  # Minimal CPU offload
```

### Step 2: Set CUDA Memory Allocation

```python
# Optimize CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256,roundup_power2_divisions:8"
```

### Step 3: Load Tokenizer

```python
from transformers import AutoTokenizer

MODEL_ID = "openai/gpt-oss-120b"
HF_TOKEN = os.environ.get("HF_TOKEN", "your_token_here")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=False,
    trust_remote_code=True,
    token=HF_TOKEN
)

# Set padding token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### Step 4: Load Model

```python
from transformers import AutoModelForCausalLM

print("Loading model (this will take 4-8 minutes)...")

# Determine attention implementation
attn_impl = "eager"  # Default
try:
    import flash_attn
    attn_impl = "flash_attention_2"
    print("✅ Using FlashAttention 2")
except ImportError:
    print("⚠️  FlashAttention not available, using eager attention")

# Load model with automatic device mapping
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    attn_implementation=attn_impl,
    token=HF_TOKEN
)

print("✅ Model loaded successfully!")

# Freeze base model parameters (only LoRA will be trainable)
for param in model.parameters():
    param.requires_grad = False
```

---

## LoRA Configuration

### Step 1: Configure LoRA Parameters

```python
from peft import LoraConfig, get_peft_model, TaskType

# LoRA hyperparameters
lora_config = LoraConfig(
    r=8,                          # LoRA rank (lower = fewer parameters)
    lora_alpha=32,                # LoRA alpha (scaling factor)
    target_modules=[              # Modules to apply LoRA to
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj"
    ],
    lora_dropout=0.05,            # Dropout for LoRA layers
    bias="none",                  # Don't train bias terms
    task_type=TaskType.CAUSAL_LM  # Task type
)
```

### Step 2: Attach LoRA to Model

```python
# Attach LoRA adapters
model = get_peft_model(model, lora_config)

# Print trainable parameters summary
def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} ({100 * trainable / total:.4f}%)")
    print(f"Total params: {total:,}")

print_trainable_parameters(model)
```

**Expected Output:**
```
Trainable params: ~6,000,000 (0.005%)
Total params: ~116,835,000,000
```

---

## Training Setup

### Step 1: Prepare Dataset

```python
from datasets import load_dataset, DatasetDict
import glob

# Load and preprocess dataset
data_files = {"train": "./data/*.json"}
ds = load_dataset("json", data_files=data_files)

# Create validation split if needed
if "validation" not in ds:
    split = ds["train"].train_test_split(test_size=0.1, seed=42)
    ds = DatasetDict({
        "train": split["train"],
        "validation": split["test"]
    })

# Preprocessing function
def preprocess_function(examples):
    # Build prompts from your data format
    prompts = []
    labels = []
    
    for item in examples:
        # Customize this based on your data format
        prompt = build_prompt(item)  # Your prompt building function
        label = extract_label(item)  # Your label extraction function
        
        # Tokenize
        enc = tokenizer(
            prompt + str(label),
            truncation=True,
            max_length=512,
            padding=False
        )
        
        # Create labels (mask prompt tokens)
        prompt_enc = tokenizer(prompt, truncation=True, max_length=512)
        prompt_len = len(prompt_enc["input_ids"])
        
        labels_list = [-100] * prompt_len + enc["input_ids"][prompt_len:] + [tokenizer.eos_token_id]
        input_ids = enc["input_ids"] + [tokenizer.eos_token_id]
        
        # Truncate if needed
        if len(input_ids) > 512:
            input_ids = input_ids[-512:]
            labels_list = labels_list[-512:]
        
        prompts.append(input_ids)
        labels.append(labels_list)
    
    return {
        "input_ids": prompts,
        "labels": labels,
        "attention_mask": [[1] * len(ids) for ids in prompts]
    }

# Apply preprocessing
ds = ds.map(preprocess_function, batched=True, remove_columns=ds["train"].column_names)
ds.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
```

### Step 2: Configure DeepSpeed (Optional but Recommended)

Create `configs/ds_config.json`:

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "round_robin_gradients": true
  },
  "gradient_clipping": 1.0,
  "steps_per_print": 50,
  "wall_clock_breakdown": false,
  "zero_allow_untested_optimizer": true
}
```

### Step 3: Configure Training Arguments

```python
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Training arguments optimized for 2x H200
training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=3,        # Batch size per GPU
    per_device_eval_batch_size=3,
    gradient_accumulation_steps=4,        # Effective batch size = 2 × 3 × 4 = 24
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_steps=2,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    eval_strategy="steps",
    eval_steps=50,
    bf16=True,                            # BF16 for H200
    bf16_full_eval=True,
    deepspeed="configs/ds_config.json",   # DeepSpeed config (if using)
    remove_unused_columns=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    report_to="none",                     # Disable wandb/tensorboard
    optim="adamw_torch_fused",            # Fused optimizer
    gradient_checkpointing=False,         # Can enable if OOM
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM, not masked LM
)
```

### Step 4: Create Trainer

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds.get("validation"),
    data_collator=data_collator,
)
```

---

## Running Training

### Method 1: Using Trainer (Single Process)

```python
# Start training
print("Starting training...")
train_result = trainer.train()

# Save final model
trainer.save_model("./checkpoints/final")
trainer.save_state()

print("Training completed!")
print(f"Training metrics: {train_result.metrics}")
```

### Method 2: Using DeepSpeed (Multi-GPU)

```bash
# Launch training with DeepSpeed
deepspeed --num_gpus=2 train_script.py
```

Or create a training script `train_script.py`:

```python
# train_script.py
from transformers import Trainer

# ... (all setup code from above) ...

if __name__ == "__main__":
    trainer.train()
    trainer.save_model("./checkpoints/final")
```

---

## Saving Checkpoints

### Save LoRA Adapters

```python
# Save only LoRA adapters (small, ~10-50MB)
model.save_pretrained("./checkpoints/lora_adapters")
tokenizer.save_pretrained("./checkpoints/lora_adapters")
```

### Load LoRA Adapters Later

```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    token=HF_TOKEN
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./checkpoints/lora_adapters")
```

### Merge LoRA with Base Model (Optional)

```python
# Merge LoRA weights into base model
merged_model = model.merge_and_unload()

# Save merged model (large, ~240GB)
merged_model.save_pretrained("./checkpoints/merged_model")
```

---

## Troubleshooting

### Out of Memory (OOM) Errors

1. **Reduce batch size**:
   ```python
   per_device_train_batch_size=2  # or 1
   gradient_accumulation_steps=8  # Increase to maintain effective batch size
   ```

2. **Enable gradient checkpointing**:
   ```python
   gradient_checkpointing=True
   ```

3. **Reduce max sequence length**:
   ```python
   max_length=256  # Instead of 512
   ```

4. **Use CPU offloading**:
   ```python
   max_memory["cpu"] = "100GiB"  # Increase CPU offload
   ```

### Slow Training

1. **Enable Flash Attention**:
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. **Increase dataloader workers**:
   ```python
   dataloader_num_workers=8
   ```

3. **Use DeepSpeed ZeRO-2** (already configured above)

### Model Download Issues

1. **Use hf_transfer for faster downloads**:
   ```bash
   pip install hf_transfer
   export HF_HUB_ENABLE_HF_TRANSFER=1
   ```

2. **Resume interrupted downloads**:
   ```python
   # HuggingFace automatically resumes interrupted downloads
   # Just re-run the model loading code
   ```

### CUDA Errors

1. **Clear GPU cache**:
   ```python
   import torch, gc
   gc.collect()
   torch.cuda.empty_cache()
   ```

2. **Check GPU availability**:
   ```bash
   nvidia-smi
   ```

3. **Restart Python kernel/session** if persistent issues occur

---

## Expected Performance

- **Model Loading**: 4-8 minutes
- **Training Speed**: ~30-45 seconds per epoch (for ~40 samples)
- **Memory Usage**: ~135GB per GPU
- **Checkpoint Size**: ~10-50MB (LoRA adapters only)

## Next Steps

After training:
1. Evaluate your model on a test set
2. Fine-tune hyperparameters (learning rate, LoRA rank, etc.)
3. Experiment with different LoRA target modules
4. Consider training for more epochs if needed

## Additional Resources

- [HuggingFace PEFT Documentation](https://huggingface.co/docs/peft)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [GPT-OSS-120B Model Card](https://huggingface.co/openai/gpt-oss-120b)

---

## License and Attribution

- This tutorial is provided as-is for educational purposes
- Ensure compliance with OpenAI's GPT-OSS-120B model license
- Respect HuggingFace's terms of service

## Support

For issues related to:
- **Model access**: Contact HuggingFace support
- **Hardware**: Contact your cloud provider (RunPod, Lambda Labs, etc.)
- **Code issues**: Check GitHub issues or create a new one

---

**Happy Fine-Tuning! 🚀**

