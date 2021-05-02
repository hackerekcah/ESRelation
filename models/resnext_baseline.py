from models import register_arch
from layers import feature
from layers.classifiers import make_fc_layers
from layers import resnet2d
import torch.nn as nn
import logging
from layers.spec_aug import freq_masking, time_masking
logger = logging.getLogger(__name__)


@register_arch
class ResNeXtBaseline(nn.Module):
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
                 **kwargs
                 ):
        super(ResNeXtBaseline, self).__init__()
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

        self.freq_mask_param = eval(freq_mask_param)
        self.time_mask_param = eval(time_mask_param)

        self.fc_layers = make_fc_layers(d_in=self.conv2d.d_out,
                                        fc_sizes=eval(fc_layers),
                                        nb_classes=nb_classes)

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

    def forward(self, x):
        x = self.feat(x)

        x = self._spec_patch_aug(x)

        x = self.conv2d(x)

        # (B, C, 1)
        x = x.unsqueeze(-1)

        for fc in self.fc_layers:
            x = fc(x)

        return x
