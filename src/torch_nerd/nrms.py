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

    def forward(self, queries, keys, values):
        queries = self.query_layer(queries)
        keys = self.key_layer(keys)
        values = self.value_layer(values)

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, values)
        return attention_output
    
class NewsEncoder(nn.Module):
    def __init__(self, hparams, word2vec_embedding, seed=None):
        super().__init__()
        self.hparams = hparams
        self.word2vec_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(word2vec_embedding), freeze=False
        )
        # Self-Attention Layer
        self.self_attention = SelfAttention(hparams.head_num, hparams.head_dim)
        # Attention Aggregation Layer
        self.attention_layer = AttentionLayer2(hparams.attention_hidden_dim)

    def forward(self, candidate_titles):

        embedded_titles = self.word2vec_embedding(candidate_titles)
        attention_output = self.self_attention(embedded_titles, embedded_titles, embedded_titles)
        news_representation = self.attention_layer(attention_output)

        return news_representation


class UserEncoder(nn.Module):
    def __init__(self, hparams, seed=None):
        super().__init__()
        self.hparams = hparams
        self.attention_layer = AttentionLayer2(hparams.attention_hidden_dim)

    def forward(self, clicked_titles):
        user_representation = self.attention_layer(clicked_titles)
        return user_representation


class ClickPredictor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, news_representation, user_representation):
        return torch.matmul(news_representation, user_representation)

class NRMSModel(nn.Module):
    def __init__(self, hparams, word2vec_embedding, seed=None):
        super().__init__()
        self.hparams = hparams
        self.word2vec_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(word2vec_embedding), freeze=False
        )

        self.news_encoder = NewsEncoder(hparams, word2vec_embedding)
        self.user_encoder = UserEncoder(hparams)
        self.click_predictor = ClickPredictor()


    def forward(self, clicked_titles, candidate_titles):
        news_representation = self.news_encoder.forward(candidate_titles) # r in the paper
        user_representation = self.user_encoder.forward(clicked_titles)   # u in the paper
        click_probability = self.click_predictor.forward(news_representation, user_representation) # y_hat in the paper
        return F.softmax(click_probability, dim=-1) # p_i in the paper


