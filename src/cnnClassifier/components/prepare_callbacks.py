import os, time, tensorflow as tf # type: ignore
from cnnClassifier.entity.config_entity import PrepareCallbacksConfig

class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config

    def _create_tb_callbacks(self):
        ts = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_root = str(self.config.tensorboard_root_log_dir)
        tb_dir = os.path.join(tb_root, f"tb_logs_at_{ts}")
        return tf.keras.callbacks.TensorBoard(log_dir=tb_dir)

    def _create_ckpt_callbacks(self):
        ckpt_path = str(self.config.checkpoint_model_filepath)  # ensure str
        if ckpt_path.endswith(("/", "\\")) or os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, "model.ckpt.keras")
        if not (ckpt_path.endswith(".keras") or ckpt_path.endswith(".h5")):
            ckpt_path += ".keras"
        return tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, save_best_only=True)

    def get_tb_ckpt_callbacks(self):
        return [self._create_tb_callbacks(), self._create_ckpt_callbacks()]
    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
    cfg = self.config.prepare_callbacks # type: ignore
    model_ckpt_dir = os.path.dirname(str(cfg.checkpoint_model_filepath))
    create_directories([model_ckpt_dir, str(cfg.tensorboard_root_log_dir)]) # type: ignore
    return PrepareCallbacksConfig(
        tensorboard_root_log_dir=str(cfg.tensorboard_root_log_dir),
        checkpoint_model_filepath=str(cfg.checkpoint_model_filepath),
    )

