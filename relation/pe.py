import torch


def get_position_encoding(xsize):
    """
    :param xsize: (B, C, F, T)
    :return: (B, 2, F, T)
    """

    bsz, csz, fsz, tsz = xsize

    # (B, 1, F, T)
    fpos = torch.linspace(start=0., end=1., steps=fsz).view(1, 1, fsz, 1).expand(bsz, -1, -1, tsz)
    # (B, 1, F, T)
    tpos = torch.linspace(start=0., end=1., steps=tsz).view(1, 1, 1, tsz).expand(bsz, -1, fsz, -1)

    return torch.cat([fpos, tpos], dim=1)


def get_relative_position_encoding(theta_xsize_2d, phi_xsize_2d):
    """
    :param theta_xsize_2d: (B, c-inter, F1, T1)
    :param phi_xsize_2d: (B, c-inter, F2, T2)
    :return: (B, 2, Ni, Nj)
    """

    # (B, 2, F1*T1, 1)
    theta_pe = get_position_encoding(theta_xsize_2d).view(theta_xsize_2d[0], 2, -1).unsqueeze(-1)
    # (B, 2, 1, F2*T2)
    phi_pe = get_position_encoding(phi_xsize_2d).view(phi_xsize_2d[0], 2, -1).unsqueeze(-2)

    return theta_pe - phi_pe