import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttLayer2(nn.Module):
    """Soft alignment attention implementation using nn.Linear."""

    def __init__(self, input_dim, attention_dim=200):
        super(AttLayer2, self).__init__()

        # Linear transformation to project inputs into attention space
        self.attention_projection = nn.Linear(input_dim, attention_dim)
        # Query vector to score each time step
        self.query_vector = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, inputs):
        """
        Args:
            inputs: Tensor of shape (batch_size, seq_len, input_dim)
                    where seq_len is the number of time steps in the sequence.
        Returns:
            Tensor of shape (batch_size, input_dim) representing the context vector.
        """
        # Project the inputs into the attention space with a non-linearity (tanh)
        # Shape: (batch_size, seq_len, attention_dim)
        attention = torch.tanh(self.attention_projection(inputs)) # b = tanh(W * h + v)

        # Compute attention scores by projecting the transformed inputs onto the query vector
        # Shape: (batch_size, seq_len, 1)
        attention_scores = self.query_vector(attention) # a = q^T * tanh(b)

        # Squeeze the last singleton dimension to get scores for each time step
        # Shape: (batch_size, seq_len)
        attention_scores = attention_scores.squeeze(-1) 

        # Normalize the scores using softmax to obtain attention weights
        # Shape: (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1) # α = softmax(a)

        # Compute the context vector as the weighted sum of the inputs
        # Shape: (batch_size, input_dim)
        context_vector = (inputs * attention_weights).sum(dim=1) # r = sum(α * h)

        return context_vector

class SelfAttention(nn.Module):
    """Multi-head self-attention implementation using nn.Linear."""
    
    def __init__(self, input_dim, head_nums, head_dim):
        super(SelfAttention, self).__init__()

        # Number of attention heads (h in the paper)
        self.head_nums = head_nums
        # Dimension of each attention head (d_k in the paper)
        self.head_dim = head_dim
        # Total output dimension = head_nums * head_dim
        self.output_dim = head_nums * head_dim

        # Linear layers for projecting query, key, and value
        self.query_proj = nn.Linear(input_dim, self.output_dim)
        self.key_proj = nn.Linear(input_dim, self.output_dim)
        self.value_proj = nn.Linear(input_dim, self.output_dim)

    def forward(self, query, key, value):
        """
        Args:
            query, key, value: Tensors of shape (batch_size, seq_len, input_dim)
        Returns:
            Tensor of shape (batch_size, seq_len, head_nums * head_dim)
        """
        # Extract the batch size and sequence length from the input shape
        batch_size, seq_len, _ = query.size()

        # Project the inputs into the attention subspace
        Q_proj = self.query_proj(query)  # (batch_size, seq_len, head_nums * head_dim)
        K_proj = self.key_proj(key)      # (batch_size, seq_len, head_nums * head_dim)
        V_proj = self.value_proj(value)  # (batch_size, seq_len, head_nums * head_dim)

        # Reshape for multi-head attention:
        # From (batch_size, seq_len, head_nums * head_dim) to (batch_size, head_nums, seq_len, head_dim)
        Q_proj = Q_proj.view(batch_size, seq_len,
                             self.head_nums, self.head_dim).permute(0, 2, 1, 3)
        K_proj = K_proj.view(batch_size, seq_len,
                             self.head_nums, self.head_dim).permute(0, 2, 1, 3)
        V_proj = V_proj.view(batch_size, seq_len,
                             self.head_nums, self.head_dim).permute(0, 2, 1, 3)

        # Compute scaled dot-product attention scores:
        # scores = Q * K^T / sqrt(d_k)
        scores = torch.matmul(
            Q_proj, K_proj.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply the softmax function to normalize scores along the last dimension
        # Shape: (batch_size, head_nums, seq_len, seq_len)
        attention_weights = F.softmax(scores, dim=-1)

        # Compute the output of the attention layer:
        # output = softmax(scores) * V
        # (batch_size, head_nums, seq_len, head_dim)
        output = torch.matmul(attention_weights, V_proj)

        # Reshape the output back to original dimensions:
        # From (batch_size, head_nums, seq_len, head_dim) to (batch_size, seq_len, head_nums * head_dim)
        output = output.permute(0, 2, 1, 3).contiguous().view(
            batch_size, seq_len, self.output_dim)

        return output

class PositionEncoder(nn.Module):
    def __init__(self, embedding_dim, seq_len, dropout):
        super(PositionEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(seq_len, embedding_dim)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)