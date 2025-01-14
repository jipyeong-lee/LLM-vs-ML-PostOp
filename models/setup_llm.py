import torch
from vllm import LLM
from config.settings import get_model_path
import os

def setup_llm():
    """LLM 모델 설정"""
    model_path = get_model_path()

    # CUDA_VISIBLE_DEVICES에서 GPU 수를 계산
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible_devices:
        gpu_count = len(cuda_visible_devices.split(","))
    else:
        gpu_count = torch.cuda.device_count()  # 환경 변수가 없으면 시스템의 전체 GPU 수를 사용

    # tensor_parallel_size를 GPU 수에 맞게 설정
    tensor_parallel_size = min(gpu_count, 4)  # 최대 4개 GPU로 제한 (필요에 따라 조정)

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=tensor_parallel_size,  # 동적으로 설정
        use_v2_block_manager=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=4096,
        max_model_len=6000,
    )
    return llm