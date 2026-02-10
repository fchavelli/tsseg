import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.register_buffer('embeddings', torch.zeros(num_embeddings, embedding_dim))
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', torch.zeros(num_embeddings, embedding_dim))
        
        # Init weights
        self.embeddings.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        self.ema_w.data.copy_(self.embeddings)
        self.ema_cluster_size.data.fill_(1)

    def forward(self, inputs):
        # inputs: [Batch, Channels, Time] -> [Batch, Time, Channels]
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embeddings).view(input_shape)
        
        # Training: Update EMA
        if self.training:
            # Cluster size update
            encodings_sum = encodings.sum(0)
            self.ema_cluster_size.data.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
            
            # Laplace smoothing of cluster size
            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            )
            
            # Weight update
            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            
            # Normalize weights
            self.embeddings.data.copy_(self.ema_w / cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return loss, quantized.permute(0, 2, 1).contiguous(), encoding_indices.view(input_shape[0], -1)

class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        # Calculate padding to maintain same sequence length
        # For even kernel sizes, this might be off by 1, so we prefer odd kernel sizes or explicit padding
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class ModernEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=4):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        self.layers = nn.ModuleList([
            ResBlock1D(hidden_dim, kernel_size=5, dilation=2**i) 
            for i in range(num_layers)
        ])
    
    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return x

class PredictiveVQTSS(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_embeddings, commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.encoder = ModernEncoder(input_dim, hidden_dim)
        self.vq = VectorQuantizerEMA(num_embeddings, hidden_dim, commitment_cost, decay=decay)
        # Predictor: z_q_t -> z_t+1 (latent space prediction)
        self.predictor = nn.Sequential(
            ResBlock1D(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, 1)
        )

    def forward(self, x):
        # x: [Batch, Channels, Time]
        z = self.encoder(x)
        vq_loss, z_q, state_indices = self.vq(z)
        z_pred = self.predictor(z_q)
        return z_pred, vq_loss, state_indices, z_q
