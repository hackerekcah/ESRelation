from torch import nn
from .scaled_softmax import ScaledSoftmax
from .perplexity_entropy import calc_relation_structure
import torch
import torch.nn.functional as F
from . import register_rblock


@register_rblock('RBlockEfficient')
class RBlockEfficient(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 sub_sample=True,
                 bn_layer=True,
                 r_structure_type="zero",
                 softmax_type='softmax',
                 **kwargs):
        super(RBlockEfficient, self).__init__()

        self.r_structure_type = r_structure_type
        self.scaled_softmax = ScaledSoftmax(softmax_type=softmax_type)

        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        bn = nn.BatchNorm2d

        self.psi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels + 1,
                        out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels + 1,
                             out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.relation_project_heads = nn.ModuleList([
            nn.Conv2d(in_channels=self.inter_channels, out_channels=1, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=self.inter_channels, out_channels=1, kernel_size=1,
                      stride=1, padding=0, bias=False)
        ])

        if sub_sample == 'max':
            pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            self.psi = nn.Sequential(self.psi, pool_layer)
            self.phi = nn.Sequential(self.phi, pool_layer)
        elif sub_sample == 'avg':
            pool_layer = nn.AvgPool2d(kernel_size=(2, 2))
            self.psi = nn.Sequential(self.psi, pool_layer)
            self.phi = nn.Sequential(self.phi, pool_layer)
        elif sub_sample == 'None' or sub_sample is None:
            print("No subsample in nl_block.")
            pass
        elif sub_sample:
            pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            self.psi = nn.Sequential(self.psi, pool_layer)
            self.phi = nn.Sequential(self.phi, pool_layer)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

        nn.init.kaiming_normal_(self.W[0].weight, mode='fan_out', nonlinearity='relu')

    def calc_relation(self, theta_x, phi_x):
        """
        :param theta_x: (b, c_inter, Ni, 1)
        :param phi_x: (b, c_inter, 1, Nj)
        :return: (b, ni, nj)
        """
        # (b, 1, Ni, Nj)
        f = F.relu(self.relation_project_heads[0](theta_x) + self.relation_project_heads[1](phi_x))
        b, _, h, w = f.size()
        f = f.view(b, h, w)

        return f

    def forward(self, x, return_r_heatmap=False):
        '''
        :param x: (b, c, t, h, w)
        :param return_r_heatmap: if True return z, nl_map, else only return z.
        :return:
        '''

        batch_size = x.size(0)

        psi_x = self.psi(x).view(batch_size, self.inter_channels, -1)
        psi_x = psi_x.permute(0, 2, 1)

        # (b, c, Ni, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
        # (b, c, 1, Nj)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

        f = self.calc_relation(theta_x, phi_x)

        f_div_C = f / float(f.size(-1))
        f_for_relation = self.scaled_softmax(f, dim=-1)

        y = torch.matmul(f_div_C, psi_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        structure_feat = calc_relation_structure(r_structure_type=self.r_structure_type,
                                                 relation_matrix=f_for_relation,
                                                 xsize=x.size())

        y = torch.cat([y, structure_feat], dim=1)
        W_y = self.W(y)
        z = W_y + x

        if return_r_heatmap:
            return z, f_div_C, structure_feat
        return z
