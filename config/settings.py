import os

def set_environment_variables():
    """환경 변수 설정"""
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    os.environ['HF_HOME'] = os.getenv('HF_HOME', '/path/to/anonymous/model')  # 익명화된 경로
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['TOKENIZERS_PARALLELISM'] = 'True'

def get_data_path():
    """데이터 경로 반환 (익명화)"""
    return os.getenv('DATA_PATH', '/path/to/anonymous/data')

def get_model_path():
    """모델 경로 반환 (익명화)"""
    return os.getenv('MODEL_PATH', '/path/to/anonymous/model')