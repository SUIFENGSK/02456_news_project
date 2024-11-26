import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer2(nn.Module):
    def __init__(self, attention_hidden_dim, seed=None):
        super().__init__()
        self.attention_weight = nn.Linear(attention_hidden_dim, 1)

    def forward(self, inputs):
        scores = self.attention_weight(inputs)
        weights = F.softmax(scores, dim=1)
        outputs = torch.sum(inputs * weights, dim=1)
        return outputs


class SelfAttention(nn.Module):
    def __init__(self, head_num, head_dim, seed=None):
        super().__init__()
        self.head_num = head_num
        self.head_dim = head_dim
        self.query_layer = nn.Linear(head_dim, head_num * head_dim)
        self.key_layer = nn.Linear(head_dim, head_num * head_dim)
        self.value_layer = nn.Linear(head_dim, head_num * head_dim)

    def forward(self, inputs):
        print(f"Inputs structure: {type(inputs)}, {inputs}")
        queries, keys, values = inputs
        queries = self.query_layer(queries)
        keys = self.key_layer(keys)
        values = self.value_layer(values)

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, values)
        return attention_output


class NRMSModel(nn.Module):
    def __init__(self, hparams, word2vec_embedding, seed=None):
        super().__init__()
        self.hparams = hparams
        self.word2vec_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(word2vec_embedding), freeze=False
        )
        self.dropout = nn.Dropout(hparams.dropout)

        # Build encoders
        self.news_encoder = self._build_news_encoder()
        self.user_encoder = self._build_user_encoder()

    def _build_news_encoder(self):
        return nn.Sequential(
            self.word2vec_embedding,
            self.dropout,
            SelfAttention(self.hparams.head_num, self.hparams.head_dim),
            self.dropout,
            AttentionLayer2(self.hparams.attention_hidden_dim),
        )

    def _build_user_encoder(self):
        return nn.Sequential(
            nn.Linear(self.hparams.title_size, self.hparams.attention_hidden_dim),
            SelfAttention(self.hparams.head_num, self.hparams.head_dim),
            AttentionLayer2(self.hparams.attention_hidden_dim),
        )

    def forward(self, clicked_titles, candidate_titles):
        print(f"clicked_titles shape: {clicked_titles.shape}, candidate_titles shape: {candidate_titles.shape}")
        user_rep = self.user_encoder(clicked_titles)
        news_rep = self.news_encoder(candidate_titles)
        scores = torch.matmul(news_rep, user_rep.unsqueeze(-1)).squeeze(-1)
        return F.softmax(scores, dim=-1)
