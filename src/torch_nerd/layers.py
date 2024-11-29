import torch
import torch.nn as nn
import torch.nn.functional as F


class AttLayer2(nn.Module):
    """Soft alignment attention implementation."""

    def __init__(self, input_dim, attention_dim=200, seed=0):
        super(AttLayer2, self).__init__()
        torch.manual_seed(seed)
        self.input_dim = input_dim
        self.attention_dim = attention_dim

        # Define parameters
        self.W = nn.Parameter(torch.empty(input_dim, attention_dim))
        self.b = nn.Parameter(torch.zeros(attention_dim))
        self.q = nn.Parameter(torch.empty(attention_dim, 1))

        # Initialize parameters
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.q)

    def forward(self, inputs):
        """
        Args:
            inputs: Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Tensor of shape (batch_size, input_dim) with weighted sum
        """
        # Compute attention scores
        attention = torch.tanh(inputs @ self.W + self.b)  # Shape: (batch_size, seq_len, attention_dim)
        attention = attention @ self.q  # Shape: (batch_size, seq_len, 1)
        attention = attention.squeeze(-1)  # Shape: (batch_size, seq_len)

        # Apply softmax to obtain normalized attention weights
        attention_weights = F.softmax(attention, dim=1).unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)

        # Compute weighted sum of inputs
        weighted_input = (inputs * attention_weights).sum(dim=1)  # Shape: (batch_size, input_dim)
        return weighted_input


class SelfAttention(nn.Module):
    """Multi-head self-attention implementation."""

    def __init__(self, input_dim, multiheads, head_dim, seed=0):
        super(SelfAttention, self).__init__()
        torch.manual_seed(seed)
        self.input_dim = input_dim
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.output_dim = multiheads * head_dim

        # Define weights
        self.WQ = nn.Parameter(torch.empty(input_dim, self.output_dim))
        self.WK = nn.Parameter(torch.empty(input_dim, self.output_dim))
        self.WV = nn.Parameter(torch.empty(input_dim, self.output_dim))

        # Initialize weights
        nn.init.xavier_uniform_(self.WQ)
        nn.init.xavier_uniform_(self.WK)
        nn.init.xavier_uniform_(self.WV)

    def forward(self, query, key, value):
        """
        Args:
            query, key, value: Tensors of shape (batch_size, seq_len, input_dim)
        Returns:
            Tensor of shape (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = query.size()

        # Project inputs to multi-head space
        Q_proj = query @ self.WQ  # Shape: (batch_size, seq_len, output_dim)
        K_proj = key @ self.WK
        V_proj = value @ self.WV

        # Reshape for multi-head attention
        Q_proj = Q_proj.view(batch_size, seq_len, self.multiheads, self.head_dim).permute(0, 2, 1, 3)
        K_proj = K_proj.view(batch_size, seq_len, self.multiheads, self.head_dim).permute(0, 2, 1, 3)
        V_proj = V_proj.view(batch_size, seq_len, self.multiheads, self.head_dim).permute(0, 2, 1, 3)

        # Compute attention scores
        scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1)) / (self.head_dim ** 0.5)  # Scaled dot-product
        
        attention_weights = F.softmax(scores, dim=-1)  # Normalize scores
        output = torch.matmul(attention_weights, V_proj)  # Shape: (batch_size, multiheads, seq_len, head_dim)

        # Reshape back to original dimensions
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.output_dim)
        return output
