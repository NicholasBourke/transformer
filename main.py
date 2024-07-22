import torch
from dataclasses import dataclass
from model import Transformer

@dataclass
class ModelConfig:
    L: int = 64             # sequence length
    K: int = 32000          # vocabulary size
    D: int = 512            # model dimension 
    n_layer: int = 6
    n_head: int = 8
    dropout: float = 0.1

cfg = ModelConfig()
model = Transformer(cfg)

x = torch.randint(cfg.K, (16, cfg.L))
y = torch.randint(cfg.K, (16, cfg.L))

p = model(x, y)
