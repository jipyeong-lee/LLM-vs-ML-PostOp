import torch
from vllm import LLM
from config.settings import get_model_path
import os

def setup_llm():
    """Set up the LLM model"""
    model_path = get_model_path()

    # Calculate the number of GPUs from CUDA_VISIBLE_DEVICES
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible_devices:
        gpu_count = len(cuda_visible_devices.split(","))
    else:
        gpu_count = torch.cuda.device_count()  # If the environment variable is not set, use the total number of GPUs in the system

    # Set tensor_parallel_size according to the number of GPUs
    tensor_parallel_size = min(gpu_count, 4)  # Limit to a maximum of 4 GPUs (adjust as needed)

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=tensor_parallel_size,  # Set dynamically
        use_v2_block_manager=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=4096,
        max_model_len=6000,
    )
    return llm
