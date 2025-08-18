import os
from typing import Optional
from dataclasses import dataclass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class MLflowConfig:
    tracking_uri: str = PROJECT_ROOT + "/mlflow_tracking"
    experiment_name: str = "finance-agent"
    artifact_location: str = PROJECT_ROOT + "/mlflow_artifacts"
    run_name: Optional[str] = None

    def __post_init__(self):
        os.makedirs(self.artifact_location, exist_ok=True)


DEFAULT_MLFLOW_CONFIG = MLflowConfig()


def get_mlflow_config() -> MLflowConfig:
    config = MLflowConfig()

    if os.getenv("MLFLOW_TRACKING_URI"):
        config.tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if os.getenv("MLFLOW_EXPERIMENT_NAME"):
        config.experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")

    if os.getenv("MLFLOW_ARTIFACT_LOCATION"):
        config.artifact_location = os.getenv("MLFLOW_ARTIFACT_LOCATION")

    return config
