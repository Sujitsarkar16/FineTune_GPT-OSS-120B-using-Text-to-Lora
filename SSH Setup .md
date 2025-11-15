# Complete Step-by-Step Guide: Configure RunPod & Fine-Tune GPT-OSS-120B

This guide walks you through setting up a RunPod environment and fine-tuning the GPT-OSS-120B model on 2x H200 GPUs using LoRA.

## Prerequisites

- RunPod account with access to 2x H200 GPU pods
- SSH key (id_ed25519 or id_rsa)
- Cursor IDE (or any code editor with SSH support)
- HuggingFace account with access to GPT-OSS-120B model
- HuggingFace access token

---

## PART 1: Setup SSH Connection

### Step 1: Locate Your SSH Key

Open PowerShell in Cursor (Terminal → New Terminal) and check for SSH keys:

```powershell
# Check if SSH key exists
ls $env:USERPROFILE\.ssh\

# Common locations:
# - id_ed25519 (recommended)
# - id_rsa
# - id_ed25519.pub (public key)
```

**If you don't have an SSH key:**

```powershell
# Generate a new SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"
# Press Enter to accept default location
# Optionally set a passphrase
```

### Step 2: Get Your RunPod Pod Information

1. Go to [RunPod Dashboard](https://www.runpod.io/console/pods)
2. Create or select a pod with **2x H200 SXM GPUs**
3. Note down:
   - **Pod IP Address** (e.g., `91.199.227.82`)
   - **SSH Port** (usually `10173` or similar)
   - **Pod ID** (for reference)

### Step 3: Test SSH Connection

Test connection to your pod:

```powershell
# Replace with your pod's IP and port
ssh -p <PORT> -i $env:USERPROFILE\.ssh\id_ed25519 root@<POD_IP>
```

**Example:**

```powershell
ssh -p 10173 -i $env:USERPROFILE\.ssh\id_ed25519 root@91.199.227.82
```

**If connection fails:**

- Verify pod is running in RunPod dashboard
- Check IP address and port are correct
- Ensure SSH key permissions are correct (Windows usually handles this automatically)
- Make sure you've added your SSH public key to RunPod

### Step 4: Configure SSH Config (Optional but Recommended)

Create/edit SSH config for easier connection:

```powershell
# Create/edit SSH config
notepad $env:USERPROFILE\.ssh\config
```

Add this content (replace with your pod details):

```text
Host runpod
    HostName <YOUR_POD_IP>
    Port <YOUR_SSH_PORT>
    User root
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

Now you can connect simply with: `ssh runpod`

---

## PART 2: Upload Files to Pod

### Step 5: Navigate to Project Directory

In Cursor's terminal (PowerShell), navigate to your project directory:

```powershell
# Navigate to your project folder
cd E:\git\finetuning
```

### Step 6: Upload Training Notebook

Upload the main training notebook:

```powershell
# Replace <PORT> and <POD_IP> with your values
scp -P <PORT> -i $env:USERPROFILE\.ssh\id_ed25519 "FineTuneGPT (1).ipynb" root@<POD_IP>:/workspace/
```

**Example:**

```powershell
scp -P 10173 -i $env:USERPROFILE\.ssh\id_ed25519 "FineTuneGPT (1).ipynb" root@91.199.227.82:/workspace/
```

### Step 7: Upload Data Files

Upload your training data:

```powershell
# Upload entire data directory
scp -P <PORT> -i $env:USERPROFILE\.ssh\id_ed25519 -r data root@<POD_IP>:/workspace/
```

**Verify upload:**

```powershell
# Connect to pod
ssh -p <PORT> -i $env:USERPROFILE\.ssh\id_ed25519 root@<POD_IP>

# Once connected, check files
cd /workspace
ls -la
ls -la data/ | head -10
```

---

## PART 3: Setup Environment on Pod

### Step 8: Connect to Pod via SSH

```powershell
ssh -p <PORT> -i $env:USERPROFILE\.ssh\id_ed25519 root@<POD_IP>
```

Or if you configured SSH config:

```powershell
ssh runpod
```

### Step 9: Navigate to Workspace

```bash
cd /workspace
pwd  # Should show: /workspace
```

### Step 10: Verify GPU Availability

```bash
# Check GPU status
nvidia-smi

# You should see 2x H200 GPUs
# Expected output shows:
# - GPU 0: NVIDIA H200 141GB
# - GPU 1: NVIDIA H200 141GB
```

**Verify GPU count:**

```bash
python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
# Should output: GPU count: 2
```

### Step 11: Check Python Environment

```bash
# Check Python version (should be 3.8+)
python3 --version

# Check if required packages are installed
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"
python3 -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
python3 -c "import peft; print(f'PEFT: {peft.__version__}')"
```

**If packages are missing**, install them:

```bash
pip install -U "transformers>=4.35.0" accelerate deepspeed bitsandbytes peft datasets sentence-transformers safetensors
```

**Optional: Install hf_transfer for faster downloads:**

```bash
pip install hf_transfer
```

### Step 12: Setup HuggingFace Token

You need a HuggingFace token to access the GPT-OSS-120B model:

1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token with **read** access
3. Set it as an environment variable:

```bash
# Set HuggingFace token (replace with your token)
export HF_TOKEN="your_huggingface_token_here"
export HUGGINGFACE_TOKEN="your_huggingface_token_here"

# Verify it's set
echo $HF_TOKEN
```

**Or login via CLI:**

```bash
huggingface-cli login
# Enter your token when prompted
```

### Step 13: Configure Cache Directory (Optional)

For large models, you may want to use a custom cache directory:

```bash
# Check available disk space
df -h

# Set custom cache (adjust path as needed)
export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="/workspace/hf_cache/transformers"
export HF_DATASETS_CACHE="/workspace/hf_cache/datasets"

# Create cache directory
mkdir -p /workspace/hf_cache
```

---

## PART 4: Run Training

### METHOD A: Using Jupyter Lab (Easiest - Recommended)

#### Step 14A: Access Jupyter Lab

1. Go to [RunPod Dashboard](https://www.runpod.io/console/pods)
2. Click on your pod
3. Under "HTTP Services", click the **"Jupyter Lab"** link
4. Jupyter Lab opens in your browser

#### Step 15A: Upload and Open Notebook

1. In Jupyter Lab, click **Upload** button (top left)
2. Select `FineTuneGPT (1).ipynb` from your local machine (or it should already be in `/workspace/`)
3. Click on `FineTuneGPT (1).ipynb` to open it

#### Step 16A: Configure HuggingFace Token in Notebook

Before running cells, you need to set your HuggingFace token. In the notebook, find the cell that sets `HF_TOKEN` and update it:

```python
HF_TOKEN = "your_huggingface_token_here"
```

Or set it as an environment variable before starting Jupyter.

#### Step 17A: Run Notebook Cells Sequentially

Run cells in order:

1. **Cell 0** (Markdown): Overview and optimizations - **Read only, no execution needed**

2. **Cell 1**: Install dependencies

   ```python
   !pip install -U "transformers>=4.35.0" accelerate deepspeed bitsandbytes peft sentence-transformers safetensors
   ```

   - **Expected time**: 2-5 minutes
   - **What it does**: Installs all required packages

3. **Cell 2**: GPU cleanup and memory management
   - **Expected time**: 10-30 seconds
   - **What it does**: Clears GPU memory, kills stale processes, resets GPUs if needed

4. **Cell 3**: Setup cache directory
   - **Expected time**: < 5 seconds
   - **What it does**: Configures HuggingFace cache to use available disk space

5. **Cell 4**: Verify accelerate installation
   - **Expected time**: < 5 seconds
   - **What it does**: Ensures accelerate is properly installed

6. **Cell 5**: Install hf_transfer (optional)

   ```python
   pip install hf_transfer
   ```

   - **Expected time**: < 1 minute
   - **What it does**: Installs faster download utility

7. **Cell 6**: Load model

   - **Expected time**: 4-8 minutes
   - **What it does**:
     - Loads GPT-OSS-120B tokenizer
     - Loads model across 2 GPUs with 4-bit quantization
     - Configures memory allocation (135GB per GPU)
   - **Important**: Make sure `HF_TOKEN` is set correctly in this cell

8. **Cell 7**: Reinstall packages (if needed)
   - **Expected time**: 2-5 minutes
   - **What it does**: Ensures all packages are up to date

9. **Cell 8**: Attach LoRA adapters

   - **Expected time**: < 1 minute
   - **What it does**:
     - Configures LoRA (rank=8, alpha=32)
     - Attaches LoRA to q_proj, k_proj, v_proj, o_proj modules
     - Freezes base model, only LoRA params are trainable

10. **Cell 9**: Load and preprocess dataset
    - **Expected time**: 1-5 minutes (depends on dataset size)
    - **What it does**:
      - Loads JSON files from `./data/*.json`
      - Builds prompts from data format
      - Tokenizes and prepares training examples
      - Creates train/validation splits

11. **Cell 10**: Setup Trainer
    - **Expected time**: < 1 minute
    - **What it does**:
      - Creates DeepSpeed config
      - Configures TrainingArguments (batch size, learning rate, etc.)
      - Creates Trainer object

12. **Cell 11**: Start training
    - **Expected time**: ~30-45 seconds per epoch (for ~40 samples)
    - **What it does**: Runs fine-tuning with LoRA
    - **Output**: Checkpoints saved to `./t2l_lora_checkpoints/`

13. **Cell 12**: DeepSpeed training script (Alternative method)
    - **Expected time**: Same as Cell 11
    - **What it does**: Creates and runs training script with DeepSpeed launcher

#### Step 18A: Monitor Training

- Watch the output in Jupyter cell for:
  - Training loss decreasing
  - Evaluation metrics (if validation split exists)
  - Checkpoint saves

- Check GPU usage:

  ```bash
  watch -n 1 nvidia-smi
  ```

- Monitor checkpoint directory:

  ```bash
  ls -lh ./t2l_lora_checkpoints/
  ```

#### Step 19A: Verify Training Completion

After training completes:

```bash
# Check saved checkpoints
ls -lh ./t2l_lora_checkpoints/

# Should see:
# - adapter_config.json
# - adapter_model.safetensors
# - checkpoint-*/ (if saved during training)
```

---

### METHOD B: Using SSH Terminal (Alternative)

#### Step 14B: Convert Notebook to Python Script

```bash
# Install jupyter if not available
pip install jupyter

# Convert notebook to script
jupyter nbconvert --to script "FineTuneGPT (1).ipynb"
```

#### Step 15B: Run Training Script

```bash
# Run the converted script
python3 "FineTuneGPT (1).py"
```

Or use DeepSpeed directly:

```bash
# If using DeepSpeed training script
deepspeed --num_gpus=2 train_script.py
```

---

## PART 5: Post-Training

### Step 20: Download Checkpoints

After training completes, download your checkpoints:

```powershell
# From your local machine (PowerShell)
# Download checkpoint directory
scp -P <PORT> -i $env:USERPROFILE\.ssh\id_ed25519 -r root@<POD_IP>:/workspace/t2l_lora_checkpoints ./
```

### Step 21: Verify Checkpoints

```bash
# On local machine or pod
ls -lh t2l_lora_checkpoints/
# Should contain:
# - adapter_config.json
# - adapter_model.safetensors
# - tokenizer files
# - checkpoint-*/ directories (if saved during training)
```

### Step 22: Test Model (Optional)

You can test the fine-tuned model using the `evaluate_model.ipynb` notebook:

1. Upload `evaluate_model.ipynb` to the pod
2. Upload test data to `data_train/` directory
3. Run evaluation notebook cells

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution:**

- Reduce `per_device_train_batch_size` from 3 to 2 or 1
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Enable gradient checkpointing: `gradient_checkpointing=True`
- Reduce `max_length` in preprocessing from 512 to 256

### Issue: Model Download Fails

**Solution:**

- Verify HuggingFace token is correct and has read access
- Check internet connection
- Try using `hf_transfer` for faster downloads
- Clear cache and retry: `rm -rf ~/.cache/huggingface/hub/models--openai--gpt-oss-120b`

### Issue: GPU Not Detected

**Solution:**

```bash
# Check NVIDIA drivers
nvidia-smi

# Verify CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# Restart pod if needed
```

### Issue: Training is Slow

**Solution:**

- Enable Flash Attention (requires compilation):

  ```bash
  pip install flash-attn --no-build-isolation
  ```

- Increase `dataloader_num_workers` to 8
- Use DeepSpeed ZeRO-2 (already configured)

### Issue: Checkpoints Not Saving

**Solution:**

- Check disk space: `df -h`
- Verify `output_dir` path exists and is writable
- Check `save_steps` is not too large for your dataset size

---

## Expected Results

### Training Metrics

- **Training Loss**: Should decrease over epochs
- **Learning Rate**: Starts at 2e-4, warms up over 2 steps
- **Effective Batch Size**: 24 (2 GPUs × 3 batch × 4 accumulation)

### Checkpoint Files

- **adapter_model.safetensors**: LoRA weights (~10-50MB)
- **adapter_config.json**: LoRA configuration
- **checkpoint-*/**: Intermediate checkpoints during training

### Performance

- **Model Loading**: 4-8 minutes
- **Training Time**: ~30-45 seconds per epoch (for ~40 samples)
- **Total Time**: ~8-12 minutes for 3 epochs
- **GPU Memory**: ~135GB per GPU

---

## Next Steps

1. **Evaluate Model**: Use `evaluate_model.ipynb` to test on validation set
2. **Fine-tune Hyperparameters**: Adjust learning rate, LoRA rank, batch size
3. **Train Longer**: Increase epochs if needed
4. **Merge Adapters**: Merge LoRA weights with base model if needed
5. **Deploy**: Use fine-tuned model for inference

---

## Additional Resources

- [RunPod Documentation](https://docs.runpod.io/)
- [HuggingFace PEFT Documentation](https://huggingface.co/docs/peft)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [GPT-OSS-120B Model Card](https://huggingface.co/openai/gpt-oss-120b)

---

## Notes

- This guide assumes you have access to GPT-OSS-120B model on HuggingFace
- Ensure compliance with OpenAI's model license
- Pod IPs and ports may change - always check RunPod dashboard
- Keep your HuggingFace token secure and never commit it to version control

---

## Happy Fine-Tuning! 🚀
