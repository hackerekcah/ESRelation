import torch
import torch.nn.functional as F
import math


def entropy(relation_matrix):
    """
    :param relation_matrix: (B, Ni, Nj)
    :return: (B, Ni)
    """
    return - torch.sum(relation_matrix * torch.log2(relation_matrix + 1e-7), dim=-1)


def perplexity(relation_matrix):
    """
    :param relation_matrix: (B, Ni, Nj)
    :return: (B, Ni)
    """
    return torch.pow(2, entropy(relation_matrix))


def structure_perplexity(relation_matrix):
    """
    :param relation_matrix: (B, Ni, Nj)
    :return: (B, Ni)
    """
    nj = relation_matrix.size(-1)
    # convert to tensor
    nj = torch.tensor(float(nj), device=relation_matrix.device)

    pp = perplexity(relation_matrix) / nj

    return pp


def structure_entropy(relation_matrix):
    """
    :param relation_matrix: (B, Ni, Nj)
    :return: (B, Ni)
    """
    nj = relation_matrix.size(-1)

    # convert to tensor
    nj = torch.tensor(float(nj), device=relation_matrix.device)

    out = entropy(relation_matrix) / torch.log2(nj)

    return out


def calc_relation_structure(r_structure_type, relation_matrix, xsize, detach=True, scale=False):
    """
    :param r_structure_type: "zero", "max", "perplexity", "entropy"
    :param relation_matrix: (B, Ni, Nj)
    :param detach: True or False
    :param xsize: (B, C, F, T)
    :param scale: if true, scale structure_feat by channel size
    :return: structure_feat (B, 1, F, T)
    """

    bsz, csz, fsz, tsz = xsize

    if detach:
        relation_matrix = relation_matrix.detach()

    if r_structure_type == "zero":
        structure_feat = torch.zeros(size=relation_matrix.size()[:-1],
                           device=relation_matrix.device)
    elif r_structure_type == "max":
        structure_feat = relation_matrix.max(dim=-1, keepdim=False)[0]
    elif r_structure_type == "perplexity":
        structure_feat = structure_perplexity(relation_matrix)
    elif r_structure_type == "entropy":
        structure_feat = structure_entropy(relation_matrix)
    elif r_structure_type == "minus_entropy":
        structure_feat = 1. - structure_entropy(relation_matrix)
    elif r_structure_type == "minus_perplexity":
        structure_feat = 1. - structure_perplexity(relation_matrix)
    else:
        raise ValueError("{} not supported.".format(r_structure_type))

    # view as (B, 1, F, T)
    structure_feat = structure_feat.view(bsz, 1, fsz, tsz)

    if scale:
        structure_feat = structure_feat / math.sqrt(csz)

    return structure_feat
