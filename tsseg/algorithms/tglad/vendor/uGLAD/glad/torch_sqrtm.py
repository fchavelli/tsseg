import torch
from torch.autograd import Function


class MatrixSquareRoot(Function):
    """Matrix square root with differentiable approximation."""

    @staticmethod
    def forward(ctx, input_tensor):
        itr_th = 10
        dim = input_tensor.shape[0]
        norm = torch.norm(input_tensor)
        Y = input_tensor / norm
        identity = torch.eye(dim, dim, device=input_tensor.device)
        Z = identity.clone()
        for _ in range(itr_th):
            T = 0.5 * (3.0 * identity - Z.mm(Y))
            Y = Y.mm(T)
            Z = T.mm(Z)
        sqrtm = Y * torch.sqrt(norm)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        itr_th = 10
        grad_input = None
        (sqrtm,) = ctx.saved_tensors
        dim = sqrtm.shape[0]
        norm = torch.norm(sqrtm)
        A = sqrtm / norm
        identity = torch.eye(dim, dim, device=sqrtm.device)
        Q = grad_output / norm
        for _ in range(itr_th):
            Q = 0.5 * (Q.mm(3.0 * identity - A.mm(A)) - A.t().mm(A.t().mm(Q) - Q.mm(A)))
            A = 0.5 * A.mm(3.0 * identity - A.mm(A))
        grad_input = 0.5 * Q
        return grad_input
