from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    sequence_length: int = 12
    prediction_horizon_60: int = 12
    prediction_horizon_120: int = 24
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    dropout_rate: float = 0.3
    l2_reg: float = 0.001

@dataclass
class DataConfig:
    features: list = None
    test_size: float = 0.2
    random_state: int = 42
    
    def __post_init__(self):
        if self.features is None:
            self.features = ['insulin', 'calories', 'steps', 'carb_input', 'glucose']