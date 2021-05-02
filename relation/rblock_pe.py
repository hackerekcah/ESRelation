from torch import nn
from .scaled_softmax import ScaledSoftmax
from .perplexity_entropy import calc_relation_structure
import torch
from . import register_rblock


@register_rblock('RBlockPE')
class RBlockPE(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 sub_sample=True,
                 bn_layer=True,
                 r_structure_type="zero",
                 softmax_type='softmax',
                 **kwargs):
        super(RBlockPE, self).__init__()

        self.r_structure_type = r_structure_type
        self.scaled_softmax = ScaledSoftmax(softmax_type=softmax_type)

        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None or self.inter_channels == "None":
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        bn = nn.BatchNorm2d
        self.bn_layer = bn_layer

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

        self.concat_project = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2 + 2, 1, 1, 1, 0, bias=False),
            nn.ReLU()
        )

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
        if self.bn_layer:
            nn.init.kaiming_normal_(self.W[0].weight, mode='fan_out', nonlinearity='relu')
        else:
            nn.init.kaiming_normal_(self.W.weight, mode='fan_out', nonlinearity='relu')


    def pos_encoding(self, x):
        """
        :param x: (B, C, F, T)
        :return: (B, C+2, F, T)
        """
        bsz, csz, fsz, tsz = x.size()
        # (B, 1, F, T)
        fpos = torch.linspace(start=0., end=1., steps=fsz).view(1, 1, fsz, 1).expand(bsz, -1, -1, tsz)
        # (B, 1, F, T)
        tpos = torch.linspace(start=0., end=1., steps=tsz).view(1, 1, 1, tsz).expand(bsz, -1, fsz, -1)

        return torch.cat([x, fpos.to(x.device), tpos.to(x.device)], dim=1)

    def forward(self, x, return_r_heatmap=False):
        '''
        :param x: (b, c, t, h, w)
        :param return_r_heatmap: if True return z, nl_map, else only return z.
        :return:
        '''
        # device = x.device
        # x = x.to(self.concat_project[0].weight.device)

        batch_size = x.size(0)

        psi_x = self.psi(x).view(batch_size, self.inter_channels, -1)
        psi_x = psi_x.permute(0, 2, 1)

        # (B, C_inter+2, F, T)
        theta_x = self.pos_encoding(self.theta(x)).view(batch_size, self.inter_channels+2, -1, 1)

        # (b, c, 1, Nj)
        phi_x = self.pos_encoding(self.phi(x)).view(batch_size, self.inter_channels+2, 1, -1)

        h = theta_x.size(2)
        w = phi_x.size(3)
        # (b, c, Ni, Nj)
        theta_x = theta_x.expand(-1, -1, -1, w)
        # (b, c, Ni, Nj)
        phi_x = phi_x.expand(-1, -1, h, -1)

        pos_delta = theta_x[:, -2:, ...] - phi_x[:, -2:, ...]
        # (b, 2c+2, Ni, Nj)
        concat_feature = torch.cat([theta_x[:, :-2, ...], phi_x[:, :-2, ...], pos_delta], dim=1)
        f = self.concat_project(concat_feature)
        b, _, h, w = f.size()
        f = f.view(b, h, w)

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

        # z = z.to(device)

        if return_r_heatmap:
            return z, f_div_C, structure_feat
        return z
