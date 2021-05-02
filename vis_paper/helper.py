import csv
import glob
import os
import argparse
import torch
import numpy as np
import ast
import cv2


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
VIS_ROOT = os.path.dirname(os.path.realpath(__file__))
CKPT_ROOT = os.path.join(ROOT, 'ckpt')


def get_label_mapper():
    meta_file = os.path.join(VIS_ROOT, "esc_int2label.csv")
    label_mapper = []
    with open(meta_file) as f:
        reader = csv.DictReader(f)
        for r in reader:
            label_mapper.append(r['category'])

    return label_mapper

# a list, ESC labels.
LABLE_MAPPER = get_label_mapper()


def select_model_cfg_from_csv(exp, **kwargs):
    """
    exp: experiment name.
    """
    exp_root = os.path.join(CKPT_ROOT, exp)
    csv_file = os.path.join(exp_root, exp + '.csv')
    print("load cfg from {}".format(csv_file))
    rows = []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for r in reader:
            not_match = False
            for k, v in kwargs.items():
                if r[k] != v:
                    not_match = True
                    break

            if not_match:
                continue

            print("selecting {}, {}".format(r['exp'], r['ckpt_prefix']))

            for k, v in r.items():
                try:
                    r[k] = ast.literal_eval(v)
                except ValueError:
                    pass

                if r[k] == 'TRUE':
                    r[k] = True

                if r[k] == 'FALSE':
                    r[k] = False
                if isinstance(r[k], list):
                    r[k] = str(r[k])
            rows.append(r)
        if len(rows) == 0:
            raise ValueError('No cfg selected.')
        args = argparse.Namespace(**rows[0])
    return args


def load_model_param(args):
    exp_root = os.path.join(CKPT_ROOT, args.exp)
    ckpts = glob.glob(os.path.join(exp_root, args.ckpt_prefix+"*.tar"))
    best_acc = 0.
    idx = 1000
    for i, ckpt in enumerate(ckpts):
        acc = ckpt.split('_')[-1][:-4]
        acc = float(acc)
        if acc > best_acc:
            idx = i

    ckpt_path = ckpts[idx]
    print("loading from {}".format(ckpt_path))
    return torch.load(ckpt_path)['model_state_dict']


def get_heat_map(spec, nl_map, center_tf):
    spec_fsz, spec_tsz = spec.shape
    Ni = nl_map.shape[0]
    ni_fsz = np.ceil(np.sqrt(Ni/2.))
    ni_tsz = int(Ni / ni_fsz)

    region_center_t = center_tf[0]
    region_center_f = center_tf[1]

    mapped_fidx = np.floor(region_center_f / float(spec_fsz) * ni_fsz)
    mapped_tidx = np.floor(region_center_t / float(spec_tsz) * ni_tsz)
    nl_map_idx = mapped_fidx * ni_tsz + mapped_tidx

    heat_vector = nl_map[int(nl_map_idx)]

    nj = heat_vector.size
    nj_fsz = int(np.ceil(np.sqrt(nj / 2.)))
    nj_tsz = int(nj / nj_fsz)

    # (Fj,Tj)
    heat_map = np.reshape(heat_vector, (1, 1, nj_fsz, nj_tsz))

    # print("nl_map:min:{}, max:{}, mean:{}".format(heat_map.min(),
    #                                               heat_map.max(),
    #                                               heat_map.mean()))
    return heat_map