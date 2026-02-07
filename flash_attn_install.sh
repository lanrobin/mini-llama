# 只为 RTX 30 系列 (Ampere) 编译
# 计算能力列表可以参考 NVIDIA 官方文档：https://developer.nvidia.com/cuda-gpus
export TORCH_CUDA_ARCH_LIST="8.6"

# RTX 50 系列 (Ada) 编译
#export TORCH_CUDA_ARCH_LIST="12.0"

# 限制并发进程数以防爆内存
# 这个数据可以根据你的系统内存和 CPU 核心数进行调整。
# 32GB 内存的系统建议设置为 4-6，64GB 内存的系统可以设置为 8-12。
export MAX_JOBS=10

# 开始安装
pip install flash-attn --no-build-isolation --no-cache-dir --force-reinstall --ignore-installed