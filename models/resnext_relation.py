import torch
from models import register_arch
from layers import feature
from layers.classifiers import make_fc_layers
from layers import resnet2d
from relation import RBLOCK_REGISTRY
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
from layers.spec_aug import freq_masking, time_masking
logger = logging.getLogger(__name__)


@register_arch
class ResNeXtRelation(nn.Module):
    def __init__(self,
                 resnet2d_arch,
                 resnet2d_pretrained,
                 resnet2d_use_num_layers,
                 resnet2d_pool,
                 feat,
                 n_mels,
                 sr,
                 fc_layers,
                 nb_classes,
                 feat_interpolate,
                 freq_mask_param,
                 time_mask_param,
                 r_stage_block,
                 r_block_name,
                 r_structure_type,
                 softmax_type,
                 **kwargs
                 ):
        super(ResNeXtRelation, self).__init__()
        feat_cls = getattr(feature, feat)
        self.feat = feat_cls(n_mels=n_mels, sr=sr,
                             feat_interpolate=feat_interpolate,
                             n_frames=kwargs.get('n_frames', 250))

        # conv2d models, using resnet 2d models
        resnet_cls = getattr(resnet2d, resnet2d_arch)
        logger.info("loading conv2d architecture {}".format(resnet2d_arch))
        self.conv2d = resnet_cls(pretrained=resnet2d_pretrained,
                                 resnet2d_pool=resnet2d_pool,
                                 use_num_layers=resnet2d_use_num_layers)

        if type(r_stage_block) is tuple:
            self.r_stage_id, self.r_block_id = r_stage_block
        else:
            self.r_stage_id, self.r_block_id = eval(r_stage_block)

        if self.r_stage_id > 0:
            d_in = 64 * self.conv2d.block_expansion * 2 ** (self.r_stage_id - 1)

            rblock_cls = RBLOCK_REGISTRY[r_block_name]
            self.rblock = rblock_cls(in_channels=d_in,
                                     r_structure_type=r_structure_type,
                                     softmax_type=softmax_type,
                                     **kwargs)
        else:
            self.rblock = None

        self.resnet2d_pool = resnet2d_pool

        self.freq_mask_param = eval(freq_mask_param)
        self.time_mask_param = eval(time_mask_param)

        self._reset_rblcok_parameters()

        self.fc_layers = make_fc_layers(d_in=self.conv2d.d_out,
                                        fc_sizes=eval(fc_layers),
                                        nb_classes=nb_classes)

    def _reset_rblcok_parameters(self):
        if self.rblock is None:
            return
        self.rblock.reset_parameters()

    def _spec_patch_aug(self, x):
        """
        apply aug sequentially.
        :param x:
        :return:
        """
        if self.training:
            x = time_masking(x, time_mask_param=self.time_mask_param)
            x = freq_masking(x, freq_mask_param=self.freq_mask_param)
        return x

    def forward_with_rblock(self, x):
        stage_id = 0
        for c in self.conv2d.children():
            if len(list(c.children())) == 0:
                x = c(x)
            else:
                stage_id += 1
                for block_id, block in enumerate(c.children()):
                    x = block(x)
                    if block_id == self.r_block_id and stage_id == self.r_stage_id:
                        x = self.rblock(x)
        return x

    def get_r_heatmap(self, x):
        feat = self.feat(x)
        x = feat
        stage_id = 0
        for c in self.conv2d.children():
            if len(list(c.children())) == 0:
                x = c(x)
            else:
                stage_id += 1
                for block_id, block in enumerate(c.children()):
                    x = block(x)
                    if block_id == self.r_block_id and stage_id == self.r_stage_id:
                        x, relation, structure_feat = self.rblock(x, return_r_heatmap=True)
                        return feat, relation, structure_feat

    def forward(self, x):
        x = self.feat(x)

        x = self._spec_patch_aug(x)

        x = self.forward_with_rblock(x)

        if self.resnet2d_pool == 'avg':
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)

        elif self.resnet2d_pool == 'max':
            x = F.adaptive_max_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)

        # (B, C, 1)
        x = x.unsqueeze(-1)

        for fc in self.fc_layers:
            x = fc(x)

        return x
