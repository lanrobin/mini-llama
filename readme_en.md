[中文版](Readme.md)

# Mini-Llama

A tutorial project for learning the inference pipeline of Large Language Models (LLMs).

## Table of Contents

- [Introduction](#introduction)
- [Runtime Environment](#runtime-environment)
- [Installation & Running](#installation--running)
- [Configuring VSCode Debugging](#configuring-vscode-debugging)
- [Using Flash Attention](#using-flash-attention)

---

## Introduction

This project is inspired by [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and implements a streamlined LLM inference engine based on [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct). The primary goal is **learning and understanding**. The following modifications were made on top of the original:

1. **Pure PyTorch Flash Attention Implementation**: Re-implemented `flash_attn_varlen_func` and `flash_attn_with_kvcache` using pure PyTorch, making it easier to understand the essence of Attention computation. The original `flash-attn` library can also be installed for higher performance (see below).
2. **Simplified KV Cache**: Re-implemented a more understandable `store_kvcache` to help readers clarify the KV Cache management logic.
3. **Extensive Chinese Comments**: Based on the author's understanding, detailed Chinese comments were added throughout the codebase to lower the barrier for engineers without a machine learning background.

---

## Runtime Environment

| Component | Recommended Configuration |
| --- | --- |
| OS | [WSL 2 + Ubuntu 24.04](https://learn.microsoft.com/en-us/windows/wsl/install) (bare-metal Linux also works) |
| GPU | NVIDIA RTX 3090 Ti (24 GB VRAM) |
| Framework | PyTorch 2.10.0+cu128 |

> **Not enough VRAM?** If you have less than 12 GB of VRAM, you can switch to [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct). The model is smaller but does not affect your learning of the project code.

> **Bare-metal Linux** should also work in theory, but you will need to manually install the NVIDIA driver, which can be slightly more involved.

### GPU Driver Notes for WSL 2

In a Windows + WSL 2 environment, you only need to install the latest NVIDIA driver on the Windows side. Linux in WSL will automatically use the host driver via GPU para-virtualization technology — **no separate driver installation is needed inside WSL**.

![Windows Driver](res/windows-nvidia-driver.png)
![Linux Driver](res/linux-nvidia-driver.png)

---

## Installation & Running

### 1. Install System Dependencies

```bash
sudo apt update
sudo apt install git build-essential python3.12-venv python3-dev
```

### 2. Create a Virtual Environment and Install HuggingFace CLI

```bash
mkdir mini-llama && cd mini-llama
python3 -m venv .venv
source .venv/bin/activate
pip3 install -U "huggingface_hub[cli]"
```

![venv setup](res/venv-setup.png)

### 3. Log in to HuggingFace

Go to [HuggingFace](https://huggingface.co/) to register an account, obtain an access token from the [Token page](https://huggingface.co/settings/tokens), then run:

```bash
hf auth login --token YOUR_TOKEN_FROM_HF
```

![hf_login](res/hf_login_token.png)

### 4. Clone the Repository & Download the Model

First, agree to the [Llama Model License Agreement](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/tree/main) on HuggingFace, then run:

```bash
git clone https://github.com/lanrobin/mini-llama.git
./mini-llama/llama3-2_3B.sh
```

Directory structure after download:

![llama-3.2-3b](res/llama-3.2-3b.png)

### 5. Install Python Dependencies and Run

```bash
pip3 install -r mini-llama/requirements.txt
python3 mini-llama/mini_llama.py
```

Starting up:
![run-llama-begin](res/run-mini-llama-1.png)

After some time, inference results are output:
![run-llama-end](res/run-mini-llama-2.png)

---

## Configuring VSCode Debugging

### 1. Install Extensions

After installing VSCode on Windows, install the **WSL** and **Python** extensions, then click the `><` icon at the bottom-left corner and select **Open a Remote Window**.

### 2. Connect to WSL

Select your WSL distribution:

![select-wsl-distro](res/select-wsl-distro.png)

### 3. Add Debug Configuration

Open the `~/mini-llama` directory, select `mini_llama.py`, then go to **Run → Add Configuration...** to add the following configuration:

![open_add_config](res/add-debugger-config.png)

```jsonc
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Current File",
            "type": "debugpy",
            "request": "launch",
            "cwd": "${workspaceFolder}",
            "program": "${workspaceFolder}/mini-llama/mini_llama.py",
            //"program": "${workspaceFolder}/mini-llama/test.py",
            "console": "integratedTerminal",
            "env": {
                "CUDA_LAUNCH_BLOCKING": "1", //Optional: Enable CUDA's synchronous error checking
                "TORCH_USE_CUDA_DSA": "1"  // Optional: Enable Device-side Assertions
            }
        }
    ]
}
```

### 4. Start Debugging

After saving the configuration, set breakpoints in the left sidebar of `mini_llama.py`, then press **F5** (or go to **Run → Start Debugging**) to start debugging.

![start_debug](res/break-point-start-debug.png)

After a moment, the code will pause at the breakpoint, and you can step through the code:

![break_point_hit](res/debug-break-point.png)

---

## Using Flash Attention

The project uses a pure PyTorch Attention implementation by default. To use the high-performance `flash-attn` library, follow these steps.

### 1. Install CUDA Toolkit 12.8

```bash
# Download NVIDIA repository pin file (configure priority)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Download CUDA 12.8 WSL-specific installer (~3 GB)
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-wsl-ubuntu-12-8-local_12.8.0-1_amd64.deb

# Install the repository package
sudo dpkg -i cuda-repo-wsl-ubuntu-12-8-local_12.8.0-1_amd64.deb

# Copy keyring (critical step, otherwise you'll get errors)
sudo cp /var/cuda-repo-wsl-ubuntu-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/

# Update apt cache
sudo apt-get update

# Install only the toolkit, not the driver (do NOT use "sudo apt-get install cuda")
sudo apt-get -y install cuda-toolkit-12-8
```

### 2. Build and Install flash-attn

```bash
export CUDA_HOME=/usr/local/cuda-12.8

# Set compute capability based on your GPU architecture
# RTX 30 series (Ampere): "8.6"
# RTX 50 series (Blackwell): "12.0"
# Full list: https://developer.nvidia.com/cuda-gpus
export TORCH_CUDA_ARCH_LIST="8.6"

# Limit concurrent compilation jobs to prevent out-of-memory issues
# For 32 GB RAM, use 4-6; for 64 GB RAM, use 8-12
export MAX_JOBS=8

# Start installation (this takes a while, please be patient)
pip3 install flash-attn --no-build-isolation --no-cache-dir --force-reinstall --ignore-installed
```

> **PyTorch version conflict?** If installation fails with an error, uninstall and reinstall the correct version first:
>
> ![flash_attn_error](res/flash-attn-error.png)
>
> ```bash
> pip3 uninstall torch torchvision torchaudio
> pip3 install -r mini-llama/requirements.txt
> python3 -c "import torch; print(torch.__version__)"  # Should output 2.10.0+cu128
> ```
>
> Then re-run the flash-attn installation command above.

Upon successful installation, you will see the following message:

![flash_attn_installed](res/flash-attn-finished.png)

### 3. Switch to flash-attn

Modify `layers/attention.py` to switch the import to the original implementation:

```python
# Original high-performance implementation (requires flash-attn installation):
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

# Pure PyTorch implementation for easier understanding:
# from layers.flash_attn_mock import flash_attn_varlen_func, flash_attn_with_kvcache
```

Then re-run:

```bash
python3 mini-llama/mini_llama.py
```
