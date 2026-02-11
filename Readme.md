[English Version](readme_en.md)

# Mini-Llama

一个用于学习大语言模型（LLM）推理流程的教学项目。

## 目录

- [项目简介](#项目简介)
- [运行环境](#运行环境)
- [安装与运行](#安装与运行)
- [配置 VSCode 调试](#配置-vscode-调试)
- [使用 Flash Attention](#使用-flash-attention)

---

## 项目简介

本项目参考 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，基于 [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) 实现了一套精简的 LLM 推理引擎。项目以**学习和理解**为首要目标，在原版基础上做了以下改动：

1. **纯 PyTorch 实现 Flash Attention**：用 PyTorch 重新实现了 `flash_attn_varlen_func` 和 `flash_attn_with_kvcache`，便于理解 Attention 的运算本质。同时也支持安装原版 `flash-attn` 以获得更高性能（详见下文）。
2. **简化版 KV Cache**：重新实现了更易理解的 `store_kvcache`，帮助读者理清 KV Cache 的管理逻辑。
3. **大量中文注释**：基于作者的理解，为代码添加了详尽的中文注释，降低非机器学习背景工程师的阅读门槛。

---

## 运行环境

| 组件 | 推荐配置 |
| --- | --- |
| 系统 | [WSL 2 + Ubuntu 24.04](https://learn.microsoft.com/en-us/windows/wsl/install)（裸机 Linux 也可） |
| GPU | NVIDIA RTX 3090 Ti（24 GB 显存） |
| 框架 | PyTorch 2.10.0+cu128 |

> **显存不足？** 如果显存少于 12 GB，可以改用 [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)，模型小但不影响对项目代码的学习。

> **裸机 Linux** 理论上也能运行，只是需要手动安装 NVIDIA 驱动，稍显麻烦。

### WSL 2 下的 GPU 驱动说明

在 Windows + WSL 2 环境中，只需在 Windows 端安装最新的 NVIDIA 驱动，WSL 中的 Linux 会通过 GPU 半虚拟化技术自动使用宿主机驱动，**无需在 WSL 内单独安装驱动**。

![Windows 驱动](res/windows-nvidia-driver.png)
![Linux 驱动](res/linux-nvidia-driver.png)

---

## 安装与运行

### 1. 安装系统依赖

```bash
sudo apt update
sudo apt install git build-essential python3.12-venv python3-dev
```

### 2. 创建虚拟环境并安装 HuggingFace CLI

```bash
mkdir mini-llama && cd mini-llama
python3 -m venv .venv
source .venv/bin/activate
pip3 install -U "huggingface_hub[cli]"
```

![venv setup](res/venv-setup.png)

### 3. 登录 HuggingFace

前往 [HuggingFace](https://huggingface.co/) 注册账号，并在 [Token 页面](https://huggingface.co/settings/tokens) 获取访问令牌，然后执行：

```bash
hf auth login --token YOUR_TOKEN_FROM_HF
```

![hf_login](res/hf_login_token.png)

### 4. 克隆代码 & 下载模型

先在 HuggingFace 上同意 [Llama 模型使用协议](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/tree/main)，然后执行：

```bash
git clone https://github.com/lanrobin/mini-llama.git
./mini-llama/llama3-2_3B.sh
```

下载完成后的目录结构：

![llama-3.2-3b](res/llama-3.2-3b.png)

### 5. 安装 Python 依赖并运行

```bash
pip3 install -r mini-llama/requirements.txt
python3 mini-llama/mini_llama.py
```

启动运行：
![run-llama-begin](res/run-mini-llama-1.png)

经过一段时间后，输出推理结果：
![run-llama-end](res/run-mini-llama-2.png)

---

## 配置 VSCode 调试

### 1. 安装插件

在 Windows 上安装 VSCode 后，安装 **WSL** 和 **Python** 插件，然后点击左下角的 `><` 图标，选择 **Open a Remote Window**。

### 2. 连接 WSL

选择你的 WSL 发行版：

![select-wsl-distro](res/select-wsl-distro.png)

### 3. 添加调试配置

打开 `~/mini-llama` 目录，选中 `mini_llama.py`，然后通过菜单 **Run → Add Configuration...** 添加如下配置：

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
                "CUDA_LAUNCH_BLOCKING": "1", //可选：启用CUDA的同步错误检查
                "TORCH_USE_CUDA_DSA": "1"  //可选：如果要启用设备端断言（Device-side Assertions）
            }
        }
    ]
}
```

### 4. 开始调试

保存配置后，在 `mini_llama.py` 左侧边栏设置断点，按 **F5**（或菜单 **Run → Start Debugging**）启动调试。

![start_debug](res/break-point-start-debug.png)

代码运行片刻后会在断点处暂停，即可单步调试：

![break_point_hit](res/debug-break-point.png)

---

## 使用 Flash Attention

项目默认使用纯 PyTorch 实现的 Attention，如需使用高性能的 `flash-attn` 库，请按以下步骤操作。

### 1. 安装 CUDA Toolkit 12.8

```bash
# 下载 NVIDIA 仓库的 Pin 文件（配置优先级）
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600

# 下载 CUDA 12.8 的 WSL 专用安装包（约 3 GB）
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-wsl-ubuntu-12-8-local_12.8.0-1_amd64.deb

# 安装仓库包
sudo dpkg -i cuda-repo-wsl-ubuntu-12-8-local_12.8.0-1_amd64.deb

# 复制 Keyring（关键步骤，否则会报错）
sudo cp /var/cuda-repo-wsl-ubuntu-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/

# 更新 apt 缓存
sudo apt-get update

# 只安装 Toolkit，不安装驱动（切勿使用 "sudo apt-get install cuda"）
sudo apt-get -y install cuda-toolkit-12-8
```

### 2. 编译安装 flash-attn

```bash
export CUDA_HOME=/usr/local/cuda-12.8

# 根据你的 GPU 架构设置计算能力
# RTX 30 系列 (Ampere): "8.6"
# RTX 50 系列 (Blackwell): "12.0"
# 完整列表参考: https://developer.nvidia.com/cuda-gpus
export TORCH_CUDA_ARCH_LIST="8.6"

# 限制并发编译数以防内存不足
# 32 GB 内存建议 4-6，64 GB 内存可设置 8-12
export MAX_JOBS=8

# 开始安装（耗时较长，请耐心等待）
pip3 install flash-attn --no-build-isolation --no-cache-dir --force-reinstall --ignore-installed
```

> **遇到 PyTorch 版本冲突？** 如果安装过程报错，请先卸载再重新安装正确版本：
>
> ![flash_attn_error](res/flash-attn-error.png)
>
> ```bash
> pip3 uninstall torch torchvision torchaudio
> pip3 install -r mini-llama/requirements.txt
> python3 -c "import torch; print(torch.__version__)"  # 应输出 2.10.0+cu128
> ```
>
> 然后重新执行上面的 flash-attn 安装命令。

安装成功后会看到如下提示：

![flash_attn_installed](res/flash-attn-finished.png)

### 3. 切换到 flash-attn

修改 `layers/attention.py`，将 import 切换为原版实现：

```python
# 原始高性能实现（需要安装 flash-attn）:
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

# 纯 PyTorch 实现，便于理解逻辑:
# from layers.flash_attn_mock import flash_attn_varlen_func, flash_attn_with_kvcache
```

然后重新运行即可：

```bash
python3 mini-llama/mini_llama.py
```
