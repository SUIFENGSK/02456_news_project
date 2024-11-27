import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import AttLayer2 as AdditiveAttention
from layers import SelfAttention as SelfAttention

class NewsEncoder(nn.Module):
    def __init__(self, hparams, word2vec_embedding, seed):
        super(NewsEncoder, self).__init__()
        self.embedding = word2vec_embedding
        self.dropout = nn.Dropout(hparams.dropout)
        self.self_attention = SelfAttention(
            hparams.head_num, hparams.head_dim, seed=seed
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(hparams.head_dim * hparams.head_num, 400),
            nn.ReLU(),
            nn.BatchNorm1d(400),
            nn.Dropout(hparams.dropout),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.BatchNorm1d(400),
            nn.Dropout(hparams.dropout),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.BatchNorm1d(400),
            nn.Dropout(hparams.dropout),
        )
        self.att_layer = AdditiveAttention(hparams.attention_hidden_dim, seed=seed)

    def forward(self, sequences_input_title):
        sequences_input_title = sequences_input_title.long()
        embedded_sequences = self.embedding(sequences_input_title)
        y = self.dropout(embedded_sequences)
        y = self.self_attention(y, y, y)
        y = y.view(-1, y.size(2) * y.size(1))  # Flatten for dense layers
        y = self.dense_layers(y)
        return self.att_layer(y)

class UserEncoder(nn.Module):
    def __init__(self, hparams, title_encoder, seed):
        super(UserEncoder, self).__init__()
        self.title_encoder = title_encoder
        self.self_attention = SelfAttention(
            hparams.head_num, hparams.head_dim, seed=seed
        )
        self.att_layer = AdditiveAttention(hparams.attention_hidden_dim, seed=seed)

    def forward(self, his_input_title):
        click_title_presents = torch.stack(
            [self.title_encoder(title) for title in his_input_title], dim=1
        )
        y = self.self_attention(click_title_presents, click_title_presents, click_title_presents)
        return self.att_layer(y)

class ClickPredictor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, news_representation, user_representation):
        return torch.matmul(news_representation, user_representation.unsqueeze(-1)).squeeze(-1)

class NRMSModel(nn.Module):
    def __init__(self, hparams, word2vec_embedding, seed=None):
        super().__init__()        
        tensor_word2vec_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(word2vec_embedding), freeze=False
        )

        self.news_encoder = NewsEncoder(hparams, tensor_word2vec_embedding, seed)
        self.user_encoder = UserEncoder(hparams, self.news_encoder, seed)
        self.click_predictor = ClickPredictor()


    def forward(self, pred_input_title, his_input_title):
        user_representation = self.user_encoder(his_input_title)  # u in the paper
        news_representations = torch.stack([self.news_encoder(title) for title in pred_input_title], dim=1)
        click_probability = self.click_predictor(news_representations, user_representation) # y_hat in the paper
        return F.softmax(click_probability, dim=-1) # p_i in the paper


