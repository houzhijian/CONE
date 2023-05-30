import torch.nn.functional as F
from torch import nn


class VisualAdapter(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, cfg):
        super(VisualAdapter, self).__init__()
        input_dim = cfg.INPUT_SIZE
        hidden_dim = cfg.HIDDEN_SIZE
        num_layers = cfg.NUM_LAYERS
        output_dim = input_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



