from dataclasses import dataclass
import os
from pathlib import Path
from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    PrepareCallbacksConfig,
    TrainingConfig,
    EvaluationConfig
)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
    ):
        # Load YAMLs
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        # Ensure artifacts root exists
        create_directories([str(self.config.artifacts_root)])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config.data_ingestion
        create_directories([str(cfg.root_dir)])
        return DataIngestionConfig(
            root_dir=str(cfg.root_dir),
            source_URL=str(cfg.source_URL),
            local_data_file=str(cfg.local_data_file),
            unzip_dir=str(cfg.unzip_dir),
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        cfg = self.config.prepare_base_model
        create_directories([str(cfg.root_dir)])
        return PrepareBaseModelConfig(
            root_dir=str(cfg.root_dir),
            base_model_path=str(cfg.base_model_path),
            updated_base_model_path=str(cfg.updated_base_model_path),
            params_image_size=list(self.params.IMAGE_SIZE),
            params_learning_rate=float(self.params.LEARNING_RATE),
            params_include_top=bool(self.params.INCLUDE_TOP),
            params_weights=str(self.params.WEIGHTS),
            params_classes=int(self.params.CLASSES),
        )

    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        cfg = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(str(cfg.checkpoint_model_filepath))
        create_directories([model_ckpt_dir, str(cfg.tensorboard_root_log_dir)])
        return PrepareCallbacksConfig(
            tensorboard_root_log_dir=str(cfg.tensorboard_root_log_dir),
            checkpoint_model_filepath=str(cfg.checkpoint_model_filepath),
        )

    def get_training_config(self) -> TrainingConfig:
        cfg = self.config.training
        create_directories([str(cfg.root_dir)])
        return TrainingConfig(
            updated_base_model_path=str(cfg.updated_base_model_path),
            training_data=str(cfg.training_data),
            trained_model_path=str(cfg.trained_model_path),
            params_epochs=int(self.params.EPOCHS),
            params_batch_size=int(self.params.BATCH_SIZE),
            params_image_size=list(self.params.IMAGE_SIZE),
            params_is_augmentation=bool(self.params.AUGMENTATION),
        )
    
    def get_validation_config(self) -> EvaluationConfig: 
        eval_config = EvaluationConfig( 
            path_of_model=Path("artifacts/training/model.h5"),
            training_data=Path("artifacts/data_ingestion/Chicken-fecal-images"),
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config
    
    

