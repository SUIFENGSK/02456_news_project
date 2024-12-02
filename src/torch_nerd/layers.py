import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttLayer2(nn.Module):
    """Soft alignment attention implementation."""

    def __init__(self, input_dim, attention_dim=200):
        super(AttLayer2, self).__init__()

        # Input dimensionality of each time step
        self.input_dim = input_dim
        # Dimensionality of the attention layer
        self.attention_dim = attention_dim

        # Define the learnable parameters:
        # W: Transformation matrix to project inputs into attention space
        self.W = nn.Parameter(torch.empty(input_dim, attention_dim))
        # b: Bias term for the attention computation
        self.b = nn.Parameter(torch.zeros(attention_dim))
        # q: Query vector to score each time step
        self.q = nn.Parameter(torch.empty(attention_dim, 1))

    def forward(self, inputs):
        """
        Args:
            inputs: Tensor of shape (batch_size, seq_len, input_dim)
                    where seq_len is the number of time steps in the sequence.
        Returns:
            Tensor of shape (batch_size, input_dim) representing the context vector.
        """
        # Project the inputs into the attention space:
        # Apply a linear transformation followed by a non-linearity (tanh)
        # inputs @ self.W: (batch_size, seq_len, attention_dim)
        # Shape: (batch_size, seq_len, attention_dim)
        attention = torch.tanh(inputs @ self.W + self.b)

        # Compute attention scores by projecting the transformed input onto the query vector (q)
        # attention @ self.q: (batch_size, seq_len, 1)
        attention = attention @ self.q  # Shape: (batch_size, seq_len, 1)

        # Remove the last singleton dimension for further processing
        attention = attention.squeeze(-1)  # Shape: (batch_size, seq_len)

        # Normalize the scores using softmax to obtain attention weights
        # These weights sum to 1 across the sequence dimension
        # Shape: (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention, dim=1).unsqueeze(-1)

        # Compute the context vector as the weighted sum of the inputs:
        # Multiply the attention weights with the inputs element-wise, then sum across the sequence dimension
        # Shape: (batch_size, input_dim)
        weighted_input = (inputs * attention_weights).sum(dim=1)

        return weighted_input


class SelfAttention(nn.Module):
    """Multi-head self-attention implementation."""

    def __init__(self, input_dim, head_nums, head_dim):
        super(SelfAttention, self).__init__()

        # Input dimension of the embedding vectors
        self.input_dim = input_dim
        # Number of attention heads (h in the paper)
        self.head_nums = head_nums
        # Dimension of each attention head (d_k in the paper)
        self.head_dim = head_dim
        # Total output dimension = head_nums * head_dim
        self.output_dim = head_nums * head_dim

        # Learnable weights for Query (W_Q), Key (W_K), and Value (W_V)
        self.WQ = nn.Parameter(torch.empty(input_dim, self.output_dim))
        self.WK = nn.Parameter(torch.empty(input_dim, self.output_dim))
        self.WV = nn.Parameter(torch.empty(input_dim, self.output_dim))

    def forward(self, query, key, value):
        """
        Args:
            query, key, value: Tensors of shape (batch_size, seq_len, input_dim)
        Returns:
            Tensor of shape (batch_size, seq_len, head_nums * head_dim)
        """
        # Extract the batch size and sequence length from the input shape
        batch_size, seq_len, _ = query.size()

        # Project the inputs into the attention subspace using WQ, WK, and WV
        Q_proj = query @ self.WQ  # (batch_size, seq_len, head_nums * head_dim)
        K_proj = key @ self.WK    # (batch_size, seq_len, head_nums * head_dim)
        V_proj = value @ self.WV  # (batch_size, seq_len, head_nums * head_dim)

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
    def __init__(self, embedding_dim, seq_len, use_learned_positions=True):
        super(PositionEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len

        # Learned positional encoding
        if use_learned_positions:
            self.position_embeddings = nn.Embedding(seq_len, embedding_dim)
        else:
            # Sinusoidal positional encoding
            self.register_buffer("position_encodings", self.sinusoidal_positional_encoding(
                seq_len, embedding_dim))

    def forward(self, word_embeddings):
        """
        Args:
            word_embeddings: Tensor of shape (batch_size, seq_len, embedding_dim)
        Returns:
            Tensor of shape (batch_size, seq_len, embedding_dim) with positional encodings added
        """
        batch_size, seq_len, embedding_dim = word_embeddings.size()

        # Ensure sequence length doesn't exceed defined maximum
        assert seq_len <= self.seq_len, "Sequence length exceeds maximum"

        if hasattr(self, "position_embeddings"):
            # Learned positional encoding
            positions = torch.arange(seq_len, device=word_embeddings.device).unsqueeze(
                0).expand(batch_size, seq_len)
            # Shape: (batch_size, seq_len, embedding_dim)
            pos_enc = self.position_embeddings(positions)
        else:
            # Sinusoidal positional encoding
            pos_enc = self.position_encodings[:seq_len, :].unsqueeze(0).expand(
                batch_size, -1, -1)  # Shape: (batch_size, seq_len, embedding_dim)

        return word_embeddings + pos_enc

    @staticmethod
    def sinusoidal_positional_encoding(seq_len, embedding_dim):
        """
        Compute sinusoidal positional encodings.
        """
        position = torch.arange(0, seq_len).unsqueeze(1)  # Shape: (seq_len, 1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -
                             (math.log(10000.0) / embedding_dim))  # Shape: (embedding_dim / 2)
        pe = torch.zeros(seq_len, embedding_dim)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
