from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SegmentationConfig:
    path_config: PathConfig
    train_config: TrainConfig
    test_config: TestConfig
    actions: ActionsConfig


@dataclass
class PathConfig:
    workspace: Path
    data_dir: Path
    pretrained_weights_path: Path
    trained_weights_file_name: str
    trained_weights_file_path: Path
    training_output_path: Path
    mapping_file_name: str
    mapping_file_path: Path
    weights_file: Path
    predictions_path: Path
    train_data_path: Path
    valid_data_path: Path
    test_data_path: Path


@dataclass
class TrainConfig:
    device: str
    epochs: int
    batch_size: int
    num_workers: int
    learning_rate: float
    pretrained_weights_path: Path
    training_output_path: Path


@dataclass
class TestConfig:
    device: str
    batch_size: int
    num_workers: int
    weights_file: Path
    predictions_dir: Path


@dataclass
class ActionsConfig:
    show_images: bool
    train: bool
    test: bool
    save_test_predictions: bool
    calculate_metrics: bool
