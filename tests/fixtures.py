"""A frozen backbone for CPU fixtures with explicitly supplied token states."""
from torch import nn
from model import BackboneOutput


class PrecomputedBackbone(nn.Module):
    def forward(self,batch):
        return BackboneOutput(batch.hidden,{})
