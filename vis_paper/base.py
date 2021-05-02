from helper import select_model_cfg_from_csv, load_model_param, LABLE_MAPPER
from models import ARCH_REGISTRY
from data import DATA_REGISTRY
from data.sampler import BalancedSampler
from torch.utils.data import DataLoader
import numpy as np
import random


class VisBase:
    """
    base class for loading trained model, feed data
    Get: spec, nl_map, relation_feat as attribute
    """
    def __init__(self,
                 exp,
                 **kwargs):

        self.set_seed(seed=0)

        self.exp = exp
        self.args = select_model_cfg_from_csv(exp, **kwargs)
        self.model = self.load_model()
        self.data_loader = list(self.get_data_loader())
        self.spec = None
        self.nl_map = None
        self.relation_feat = None
        self.label = None

        self._batch_id = -1
        self.bx = None
        self.by = None

    def set_seed(self, seed=0):
        np.random.seed(seed)
        random.seed(seed)

    def reload(self, exp, **kwargs):
        self.set_seed(0)
        kwargs['lr_pct_start'] = '0.08'
        kwargs['run_epochs'] = '70'
        kwargs['fold'] = '4'
        self.args = select_model_cfg_from_csv(exp, **kwargs)
        self.model = self.load_model()
        self.data_loader = self.get_data_loader()

    def feed(self, batch_id=0, data_id=0, target_id=None):
        if self._batch_id != batch_id:
            for bid, (bx, by) in enumerate(self.data_loader):
                if bid == batch_id:
                    self.bx = bx
                    self.by = by
                    self._batch_id = batch_id
                    break
        if target_id is None:
            x = self.bx[data_id]
            y = int(self.by[data_id].numpy())
        else:
            # (N,)
            x = self.bx[self.by == target_id].squeeze()
            y = target_id

        spec, nl_map, relation_feat = self.model.get_r_heatmap(x.unsqueeze(0).to("cuda"))

        spec = spec.cpu().detach().numpy()[0, 0, ...]
        spec_norm = (spec - np.min(spec)) / (np.max(spec) - np.min(spec))
        self.spec = spec_norm
        self.nl_map = nl_map[0].cpu().detach().numpy()
        # TODO, get nl_map not divide by J
        self.nl_map *= self.nl_map.shape[1]
        self.relation_feat = relation_feat.cpu().detach().numpy()
        self.label = LABLE_MAPPER[y]

    def load_model(self):
        model_cls = ARCH_REGISTRY.get(self.args.net)
        model = model_cls(**vars(self.args)).to('cuda')
        model.load_state_dict(load_model_param(self.args))
        model.eval()
        return model

    def get_data_loader(self):
        dataset_cls = DATA_REGISTRY.get(self.args.dataset)
        train_set = dataset_cls(fold=self.args.fold, split='train', target_sr=self.args.sr)
        train_loader = DataLoader(dataset=train_set,
                                  sampler=BalancedSampler(dataset=train_set,
                                                          labels=train_set.get_train_labels(),
                                                          shuffle=False),
                                  batch_size=self.args.batch_size, drop_last=True, num_workers=1)
        return train_loader
