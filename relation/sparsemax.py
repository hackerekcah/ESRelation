# adapted from http://nlp.seas.harvard.edu/pytorch-struct/_modules/torch_struct/semirings/sparse_max.html
import torch


class SparseMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim, z=1):
        w_star = project_simplex(input, dim)
        ctx.save_for_backward(input, w_star.clone(), torch.tensor(dim))
        return w_star

    def backward(ctx, grad_output):
        input, w_star, dim = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = sparsemax_grad(grad_output, w_star, dim.item())
        return grad_input, None, None


def project_simplex(v, dim, z=1):
    """
    :param v:
    :param dim: For now, dim must be the last dimension
    :param z:
    :return:
    """
    v_sorted, _ = torch.sort(v, dim=dim, descending=True)
    cssv = torch.cumsum(v_sorted, dim=dim) - z
    ind = torch.arange(1, 1 + v.shape[dim]).to(dtype=v.dtype, device=v.device)
    cond = v_sorted - cssv / ind >= 0
    k = cond.sum(dim=dim, keepdim=True)
    tau = cssv.gather(dim, k - 1) / k.to(dtype=v.dtype)
    w = torch.clamp(v - tau, min=0)
    return w


def sparsemax_grad(dout, w_star, dim):
    out = dout.clone()
    supp = w_star > 0
    out[w_star <= 0] = 0
    nnz = supp.to(dtype=dout.dtype).sum(dim=dim, keepdim=True)
    out = out - (out.sum(dim=dim, keepdim=True) / nnz)
    out[w_star <= 0] = 0
    return out


def sparse_max(x, dim=-1):
    return SparseMax.apply(x, dim)


if __name__ == '__main__':

    from torch.autograd import gradcheck

    input = (torch.randn(2, 3, 4, dtype=torch.double, requires_grad=True), 2)
    test = gradcheck(sparse_max, input, eps=1e-6, atol=1e-4)
    print(test)
