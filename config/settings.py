import os

def set_environment_variables():
    """Set environment variables"""
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    os.environ['HF_HOME'] = os.getenv('HF_HOME', '/path/to/anonymous/model')  # Anonymized path
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['TOKENIZERS_PARALLELISM'] = 'True'

def get_data_path():
    """Return data path (anonymized)"""
    return os.getenv('DATA_PATH', '/path/to/anonymous/data')

def get_model_path():
    """Return model path (anonymized)"""
    return os.getenv('MODEL_PATH', '/path/to/anonymous/model')
