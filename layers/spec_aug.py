import torch
import torchaudio


def freq_masking(x, freq_mask_param):
    """
    see torchaudio.
    https://pytorch.org/audio/functional.html#mask-along-axis-iid
    :param x: (B, 3, F, T), B here is patch num.
    :param freq_mask_param: [mask_prob, n_mask_freq]
    :return:
    """
    with torch.no_grad():
        bsz, csz, fsz, tsz = x.size()
        mask_prob, n_mask_freq = freq_mask_param
        if mask_prob == 0:
            return x
        x_masked = torchaudio.functional.mask_along_axis_iid(x.clone(), n_mask_freq, 0., 2)
        if mask_prob == 1:
            return x_masked
        # select the whole masked patch if true, else use original patch.
        select = torch.rand((bsz,)) <= mask_prob

        select = select.view(-1, 1, 1, 1).expand([-1, csz, fsz, tsz]).to('cuda')

        x = torch.where(select, x_masked, x)
    return x.requires_grad_(False)


def time_masking(x, time_mask_param):
    """
    see torchaudio.
    https://pytorch.org/audio/functional.html#mask-along-axis-iid
    :param x: (B, 3, F, T), B here is patch num.
    :param time_mask_param: [mask_prob, n_mask_time]
    :return:
    """
    with torch.no_grad():
        bsz, csz, fsz, tsz = x.size()
        mask_prob, n_mask_time = time_mask_param
        if mask_prob == 0:
            return x
        x_masked = torchaudio.functional.mask_along_axis_iid(x.clone(), n_mask_time, 0., 3)
        if mask_prob == 1:
            return x_masked
        # select the whole masked patch if true, else use original patch.
        select = torch.rand((bsz,)) <= mask_prob

        select = select.view(-1, 1, 1, 1).expand([-1, csz, fsz, tsz]).to('cuda')
        x = torch.where(select, x_masked, x)

    return x.requires_grad_(False)


if __name__ == '__main__':
    spec = torch.ones((100, 3, 64, 5))
    fm = freq_masking(spec, [0.5, 5])
    tm = time_masking(spec, [0.1, 3])
    print()