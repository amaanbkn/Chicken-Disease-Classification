from dataclasses import dataclass
from typing import List

@dataclass
class DataIngestionConfig:
    root_dir: str
    source_URL: str
    local_data_file: str
    unzip_dir: str

@dataclass
class PrepareBaseModelConfig:
    root_dir: str
    base_model_path: str
    updated_base_model_path: str
    params_image_size: List[int]
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int

@dataclass
class PrepareCallbacksConfig:
    tensorboard_root_log_dir: str
    checkpoint_model_filepath: str

@dataclass
class TrainingConfig:
    updated_base_model_path: str
    training_data: str
    trained_model_path: str
    params_epochs: int
    params_batch_size: int
    params_image_size: List[int]
    params_is_augmentation: bool
