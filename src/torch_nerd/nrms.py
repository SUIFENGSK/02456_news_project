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
            nn.Linear(12000, 400), # TODO: remove hardcoded value
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

        self.debug = True

    def forward(self, sequences_input_title):
        if self.debug:
            print("NE0: Shape of input:", sequences_input_title.shape, ". Should be (batch_size, history_size, title_size)")
        sequences_input_title = sequences_input_title.long()

        if self.debug:
            print("NE2: Shape after casting to long:", sequences_input_title.shape, ". Should be (batch_size, history_size, title_size)")
        embedded_sequences = self.embedding(sequences_input_title)
        embedded_sequences = embedded_sequences.view(-1, embedded_sequences.size(2), embedded_sequences.size(3))

        if self.debug:
            print("NE3: Shape after embedding:", embedded_sequences.shape, ". Should be (batch_size, history_size, embedding_dim)")
        y = self.dropout(embedded_sequences)

        if self.debug:
            print("NE4: Shape after dropout:", y.shape)
        y = self.self_attention(y, y, y)

        if self.debug:
            print("NE5: Shape after self attention:", y.shape)
        y = y.view(-1, y.size(2) * y.size(1))  # Flatten for dense layers

        if self.debug:
            print("NE6: Shape after flattening:", y.shape)
        y = self.dense_layers(y)

        if self.debug:
            print("NE7: Shape after dense layers:", y.shape)
        y = self.att_layer(y)

        if self.debug:
            print("NE8: Shape after att layer:", y.shape)

        return y

class UserEncoder(nn.Module):
    def __init__(self, hparams, title_encoder, seed):
        super(UserEncoder, self).__init__()
        self.title_encoder = title_encoder
        self.self_attention = SelfAttention(
            hparams.head_num, hparams.head_dim, seed=seed
        )
        self.att_layer = AdditiveAttention(hparams.attention_hidden_dim, seed=seed)

        self.debug = True

    def forward(self, his_input_title):
        if self.debug:
            print("UE1: Shape of input:", his_input_title.shape)
        click_title_presents = self.title_encoder(his_input_title)

        if self.debug:
            print("UE2: Shape after title encoder:", click_title_presents.shape)
        y = self.self_attention(click_title_presents, click_title_presents, click_title_presents)
        
        if self.debug:
            print("UE3: Shape after self attention ", y.shape)
        y = self.att_layer(y) 
        
        if self.debug:
            print("UE4: Shape after att layer ", y.shape)
        return y

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

        self.debug = True


    def forward(self, pred_input_title, his_input_title):
        if self.debug:
            print("Model: Shape of pred_input_title:", pred_input_title.shape)
            print("Model: Shape of his_input_title:", his_input_title.shape)
        user_representation = self.user_encoder(his_input_title)  # u in the paper
        news_representations = torch.stack([self.news_encoder(title) for title in pred_input_title], dim=1)
        if self.debug:
            print("Model: Shape of user_representation:", user_representation.shape)
            print("Model: Shape of news_representations:", news_representations.shape)
        click_probability = self.click_predictor(news_representations, user_representation) # y_hat in the paper
        return F.softmax(click_probability, dim=-1) # p_i in the paper


