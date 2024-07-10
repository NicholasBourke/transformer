import torch
from torch.nn.functional import one_hot, relu, log_softmax
from dataclasses import dataclass

from model import Transformer



@dataclass
class ModelConfig:
    L: int = 512
    K: int = 256
    D: int = 512
    n_layer: int = 12
    n_head: int = 2
    dropout: float = 0.2
    bias: bool = False


cfg = ModelConfig()

model = Transformer(cfg)
