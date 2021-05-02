import torch
import torch.nn as nn
import torch.nn.functional as F
from .sparsemax import sparse_max


class ScaledSoftmax(nn.Module):
    def __init__(self, tau=1.0, softmax_type='softmax'):
        """
        :param tau: positive float temperature or 'learn'
        :param softmax_type: 'softmax', 'gumbel', 'sparsemax', 'none'
        """
        super(ScaledSoftmax, self).__init__()
        if tau == 'learn':
            self.scale = nn.Parameter(torch.ones(1), requires_grad=True)
        else:
            assert tau > 0.
            self.register_buffer('scale', torch.sqrt(torch.tensor(tau, dtype=torch.float)))
        self.softmax_type = softmax_type

    def forward(self, x, dim=-1):
        """
        :param x: (B, Ni, Nj)
        :param dim:
        :return:
        """
        if self.softmax_type == 'gumbel':
            return F.gumbel_softmax(x / (self.scale ** 2), dim=dim)
        elif self.softmax_type == 'softmax':
            return F.softmax(x / (self.scale ** 2), dim=dim)
        elif self.softmax_type == 'sparsemax':
            return sparse_max(x / (self.scale ** 2), dim=dim)
        elif self.softmax_type == 'none':
            return x / float(x.size(-1))
        elif self.softmax_type == 'laplace':
            d = x / torch.sum(x, dim=dim, keepdim=True)
            d[d.isnan()] = 0.
            return d
        else:
            raise ValueError("softmax_type: {} not supported.".format(self.softmax_type))