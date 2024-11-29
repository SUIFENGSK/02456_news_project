import torch
import torch.nn as nn
import torch.nn.functional as F


class AttLayer2(nn.Module):
    """Soft alignment attention implementation."""

    def __init__(self, dim=200, seed=0):
        super(AttLayer2, self).__init__()
        torch.manual_seed(seed)
        self.dim = dim
        self.W = None
        self.b = None
        self.q = None

    def build(self, input_dim):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.W = nn.Parameter(torch.empty(input_dim, self.dim, device=device))
        self.b = nn.Parameter(torch.zeros(self.dim, device=device))
        self.q = nn.Parameter(torch.empty(self.dim, 1, device=device))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.q)

    def forward(self, inputs):
        if self.W is None or self.b is None or self.q is None:
            input_dim = inputs.size(-1)
            self.build(input_dim)

        attention = torch.tanh(torch.matmul(inputs, self.W) + self.b)
        attention = torch.matmul(attention, self.q).squeeze(-1)

        attention = attention.exp()

        attention_weight = attention / (attention.sum(dim=0, keepdim=True) + 1e-8)
        attention_weight = attention_weight.unsqueeze(-1)
        weighted_input = inputs * attention_weight
        return weighted_input.sum(dim=1)


class SelfAttention(nn.Module):
    """Multi-head self-attention implementation."""

    def __init__(self, multiheads, head_dim, seed=0, mask_right=False):
        super(SelfAttention, self).__init__()
        torch.manual_seed(seed)
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.output_dim = multiheads * head_dim
        self.mask_right = mask_right

        self.WQ = None
        self.WK = None
        self.WV = None

    def build(self, input_dim_q, input_dim_k, input_dim_v):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.WQ = nn.Parameter(torch.empty(input_dim_q, self.output_dim, device=device))
        self.WK = nn.Parameter(torch.empty(input_dim_k, self.output_dim, device=device))
        self.WV = nn.Parameter(torch.empty(input_dim_v, self.output_dim, device=device))
        nn.init.xavier_uniform_(self.WQ)
        nn.init.xavier_uniform_(self.WK)
        nn.init.xavier_uniform_(self.WV)

    def forward(self, query, key, value):        
        # Initialize weights if they are None
        if self.WQ is None or self.WK is None or self.WV is None:
            input_dim_q = query.size(-1)
            input_dim_k = key.size(-1)
            input_dim_v = value.size(-1)
            self.build(input_dim_q, input_dim_k, input_dim_v)

        batch_size, seq_len, _ = query.shape
        Q_proj = query @ self.WQ
        K_proj = key @ self.WK
        V_proj = value @ self.WV

        Q_proj = Q_proj.view(batch_size, seq_len, self.multiheads, self.head_dim).permute(0, 2, 1, 3)
        K_proj = K_proj.view(batch_size, seq_len, self.multiheads, self.head_dim).permute(0, 2, 1, 3)
        V_proj = V_proj.view(batch_size, seq_len, self.multiheads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1)) / self.head_dim ** 0.5

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V_proj)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        return output
