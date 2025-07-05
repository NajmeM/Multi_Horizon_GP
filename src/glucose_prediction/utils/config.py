"""
Configuration classes and utilities.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model architecture and training."""
    sequence_length: int = 12
    prediction_horizon_60: int = 12
    prediction_horizon_120: int = 24
    lstm_units_60: int = 64
    lstm_units_120: int = 64
    cnn_filters: int = 64
    dropout_rate: float = 0.3
    l2_reg: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    patience: int = 10


@dataclass
class DataConfig:
    """Configuration for data processing."""
    features: List[str] = field(default_factory=lambda: [
        'insulin', 'calories', 'steps', 'carb_input', 'glucose'
    ])
    test_size: float = 0.2
    random_state: int = 42
    validation_split: float = 0.2


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    save_plots: bool = True
    plot_format: str = 'png'
    plot_dpi: int = 300
    clarke_analysis: bool = True
    generate_reports: bool = True


@dataclass
class PathConfig:
    """Configuration for file paths."""
    data_dir: str = "./data"
    models_dir: str = "./models"
    results_dir: str = "./results"
    plots_dir: str = "./plots"
    logs_dir: str = "./logs"


@dataclass
class ExperimentConfig:
    """Master configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            return cls(
                model=ModelConfig(**config_dict.get('model', {})),
                data=DataConfig(**config_dict.get('data', {})),
                evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
                paths=PathConfig(**config_dict.get('paths', {}))
            )
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            raise
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        try:
            config_dict = {
                'model': self.model.__dict__,
                'data': self.data.__dict__,
                'evaluation': self.evaluation.__dict__,
                'paths': self.paths.__dict__
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {e}")
            raise


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )