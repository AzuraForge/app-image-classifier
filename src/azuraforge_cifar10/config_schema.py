# app-image-classifier/src/azuraforge_cifar10/config_schema.py
from pydantic import BaseModel, Field
from typing import Literal, Optional

class DataSourcingConfig(BaseModel):
    train_limit: int = Field(1000, gt=0, description="Maximum number of training examples to use for quick training.")
    test_limit: int = Field(200, gt=0, description="Number of test examples to use for model evaluation.")

class TrainingParamsConfig(BaseModel):
    epochs: int = Field(10, gt=0)
    lr: float = Field(0.001, gt=0, description="Learning rate.")

class Cifar10Config(BaseModel):
    pipeline_name: Literal['cifar10_classifier']
    data_sourcing: DataSourcingConfig
    training_params: TrainingParamsConfig

    class Config:
        extra = 'forbid'