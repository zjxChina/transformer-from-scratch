from dataclasses import dataclass


@dataclass
class ModelConfig:
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 3
    d_ff: int = 1024
    dropout: float = 0.1
    use_positional_encoding: bool = True  # 新增
    use_residual: bool = True  # 新增


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 0.0003
    num_epochs: int = 50
    warmup_steps: int = 4000
    max_seq_length: int = 128
    gradient_clip: float = 1.0
    weight_decay: float = 0.01


@dataclass
class DataConfig:
    dataset_name: str = "iwslt2017"
    source_lang: str = "en"
    target_lang: str = "de"
    max_vocab_size: int = 10000


@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig

    def __init__(self, **kwargs):
        # 模型配置
        model_config = kwargs.get('model', {})
        self.model = ModelConfig(
            d_model=model_config.get('d_model', 256),
            num_heads=model_config.get('num_heads', 8),
            num_layers=model_config.get('num_layers', 3),
            d_ff=model_config.get('d_ff', 1024),
            dropout=model_config.get('dropout', 0.1),
            use_positional_encoding=model_config.get('use_positional_encoding', True),
            use_residual=model_config.get('use_residual', True)
        )

        # 训练配置
        training_config = kwargs.get('training', {})
        self.training = TrainingConfig(
            batch_size=training_config.get('batch_size', 32),
            learning_rate=training_config.get('learning_rate', 0.0003),
            num_epochs=training_config.get('num_epochs', 50),
            warmup_steps=training_config.get('warmup_steps', 4000),
            max_seq_length=training_config.get('max_seq_length', 128)
        )

        # 数据配置
        data_config = kwargs.get('data', {})
        self.data = DataConfig(
            dataset_name=data_config.get('dataset_name', "iwslt2017"),
            source_lang=data_config.get('source_lang', "en"),
            target_lang=data_config.get('target_lang', "de"),
            max_vocab_size=data_config.get('max_vocab_size', 10000)
        )
