import torch

from .torch_sqrtm import MatrixSquareRoot


torch_sqrtm = MatrixSquareRoot.apply


def get_optimizers(model_glad, lr_glad=0.002, use_optimizer="adam"):
    """Return the optimizer used for the GLAD parameters."""
    if use_optimizer == "adam":
        optimizer_glad = torch.optim.Adam(
            model_glad.parameters(),
            lr=lr_glad,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
    else:
        raise ValueError("Optimizer not found")
    return optimizer_glad


def batch_matrix_sqrt(A):
    """Matrix square root for batched positive semi-definite matrices."""
    if len(A.shape) == 2:
        return torch_sqrtm(A)
    n = A.shape[0]
    sqrtm_torch = torch.zeros_like(A)
    for i in range(n):
        sqrtm_torch[i] = torch_sqrtm(A[i])
    return sqrtm_torch


def get_frobenius_norm(A, single=False):
    """Compute Frobenius norm for a matrix or batch of matrices."""
    if single:
        return torch.sum(A ** 2)
    return torch.mean(torch.sum(A ** 2, (1, 2)))


def glad(Sb, model, lambda_init=1, L=15, INIT_DIAG=0, USE_CUDA=False):
    """Unroll the GLAD iterations to estimate the precision matrix."""
    D = Sb.shape[-1]
    if len(Sb.shape) == 2:
        Sb = Sb.reshape(1, Sb.shape[0], Sb.shape[1])
    if INIT_DIAG == 1:
        batch_diags = 1 / (
            torch.diagonal(Sb, offset=0, dim1=-2, dim2=-1) + model.theta_init_offset
        )
        theta_init = torch.diag_embed(batch_diags)
    else:
        theta_init = torch.inverse(
            Sb
            + model.theta_init_offset
            * torch.eye(D).expand_as(Sb).type_as(Sb)
        )

    theta_pred = theta_init
    identity_mat = torch.eye(Sb.shape[-1]).expand_as(Sb)

    if USE_CUDA:
        identity_mat = identity_mat.cuda()

    zero = torch.tensor([0.0])
    dtype = torch.FloatTensor
    if USE_CUDA:
        zero = zero.cuda()
        dtype = torch.cuda.FloatTensor

    lambda_k = model.lambda_forward(zero + lambda_init, zero, k=0)
    for k in range(L):
        b = 1.0 / lambda_k * Sb - theta_pred
        b2_4ac = torch.matmul(b.transpose(-1, -2), b) + 4.0 / lambda_k * identity_mat
        sqrt_term = batch_matrix_sqrt(b2_4ac)
        theta_k1 = 1.0 / 2 * (-1 * b + sqrt_term)
        theta_pred = model.eta_forward(theta_k1, Sb, k, theta_pred)
        lambda_k = model.lambda_forward(
            torch.tensor([get_frobenius_norm(theta_pred - theta_k1).item()], dtype=torch.float32).type(dtype),
            lambda_k,
            k,
        )
    return theta_pred
