import torch
import torch.nn as nn


class glad_params(torch.nn.Module):
    """Parameterization for the GLAD optimizer used within uGLAD."""

    def __init__(self, theta_init_offset, nF, H, USE_CUDA=False):
        """Initialize the GLAD meta-parameters."""
        super(glad_params, self).__init__()
        self.dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
        self.theta_init_offset = nn.Parameter(
            torch.tensor([theta_init_offset], dtype=torch.float32).type(self.dtype)
        )
        self.nF = nF  # number of input features
        self.H = H  # hidden layer size
        self.rho_l1 = self.rhoNN()
        self.lambda_f = self.lambdaNN()
        self.zero = torch.tensor([0.0], dtype=torch.float32).type(self.dtype)

    def rhoNN(self):
        """Build the rho thresholding network."""
        l1 = nn.Linear(self.nF, self.H).type(self.dtype)
        lH1 = nn.Linear(self.H, self.H).type(self.dtype)
        l2 = nn.Linear(self.H, 1).type(self.dtype)
        return nn.Sequential(l1, nn.Tanh(), lH1, nn.Tanh(), l2, nn.Sigmoid()).type(self.dtype)

    def lambdaNN(self):
        """Build the lambda update network."""
        l1 = nn.Linear(2, self.H).type(self.dtype)
        l2 = nn.Linear(self.H, 1).type(self.dtype)
        return nn.Sequential(l1, nn.Tanh(), l2, nn.Sigmoid()).type(self.dtype)

    def eta_forward(self, X, S, k, F3=None):
        """Entrywise soft-thresholding step for GLAD."""
        if F3 is None:
            F3 = []
        batch_size, shape1, shape2 = X.shape
        Xr = X.reshape(batch_size, -1, 1)
        Sr = S.reshape(batch_size, -1, 1)
        feature_vector = torch.cat((Xr, Sr), -1)
        if len(F3) > 0:
            F3r = F3.reshape(batch_size, -1, 1)
            feature_vector = torch.cat((feature_vector, F3r), -1)
        rho_val = self.rho_l1(feature_vector).reshape(X.shape)
        return torch.sign(X) * torch.max(self.zero, torch.abs(X) - rho_val)

    def lambda_forward(self, normF, prev_lambda, k=0):
        """Update lambda given the Frobenius norm of the iterate."""
        if torch.is_tensor(normF):
            normF_val = normF.detach().cpu().item()
        else:
            normF_val = float(normF)
        if torch.is_tensor(prev_lambda):
            prev_lambda_val = prev_lambda.detach().cpu().item()
        else:
            prev_lambda_val = float(prev_lambda)
        feature_vector = torch.tensor([normF_val, prev_lambda_val], dtype=torch.float32).type(self.dtype)
        return self.lambda_f(feature_vector)
