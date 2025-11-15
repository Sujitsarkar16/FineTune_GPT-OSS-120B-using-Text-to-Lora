# GPT-OSS-120B Fine-Tuning Repository

This repository contains code and resources for fine-tuning the GPT-OSS-120B model using LoRA (Low-Rank Adaptation) on 2x NVIDIA H200 SXM GPUs.

## Overview

This project implements efficient fine-tuning of the 120-billion parameter GPT-OSS model using:

- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning that trains only a small subset of model parameters
- **4-bit Quantization**: Reduces memory footprint using bitsandbytes
- **DeepSpeed ZeRO-2**: Distributed training optimization for multi-GPU setups
- **BF16 Mixed Precision**: Optimized for H200 GPUs

## Repository Structure

```text
.
├── FineTuneGPT.ipynb    # Main training notebook
├── data/                     # Training data (JSON format)
├── data_train/               # Test/evaluation data (Markdown format)
├── t2l_lora_checkpoints/     # Saved LoRA adapter checkpoints
├── offload/                  # CPU offload directory (temporary)
├── configs/                  # DeepSpeed configuration files
├── FINETUNIG_TUTORIAL.md     # DeepSpeed configuration files
└── SSH_Setup.md     # RunPod setup instructions
```

## Key Features

### Hardware Optimization

- Optimized for **2x NVIDIA H200 141GB SXM GPUs** (282GB total)
- Memory allocation: 135GB per GPU (6GB headroom)
- Minimal CPU offloading for maximum GPU utilization

### Training Configuration

- **Batch size**: 3 per device
- **Gradient accumulation**: 4 steps
- **Effective batch size**: 24 (2 GPUs × 3 × 4)
- **Learning rate**: 2e-4
- **LoRA rank**: 8, alpha: 32
- **Target modules**: q_proj, k_proj, v_proj, o_proj

### Performance

- Model loading: 4-8 minutes
- Training time: ~30-45 seconds per epoch (for 38 samples)
- Total training time: ~8-12 minutes (3 epochs)

## Data Format

Training data is provided in JSON format with the following structure:

- `score`: Numeric score for the submission
- `max_score`: Maximum possible score
- `rubric_alignment`: List of question-answer pairs with individual scores
- `feedback`: Optional feedback text

## Usage

1. **Setup Environment**: Install required dependencies (see tutorial)
2. **Prepare Data**: Format your training data as JSON files
3. **Run Training**: Execute the training notebook cells sequentially
4. **Evaluate**: Use the evaluation notebook to test the fine-tuned model

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- Transformers >= 4.35.0
- Accelerate
- DeepSpeed
- bitsandbytes
- PEFT
- Datasets

## Notes

- This setup requires significant GPU memory (2x H200 or equivalent)
- The model uses pre-quantized weights (MXFP4) from HuggingFace
- LoRA adapters are saved separately and can be merged with the base model
- Training is optimized for small to medium-sized datasets

## License

Please refer to the license of the base GPT-OSS-120B model and ensure compliance with OpenAI's usage terms.
