import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import AttLayer2 as AdditiveAttention
from layers import SelfAttention as SelfAttention
from layers import PositionEncoder as PositionEncoder


class NewsEncoder(nn.Module):
    def __init__(self, hparams, word2vec_embedding):
        super(NewsEncoder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word2vec_embedding))
        self.dropout = nn.Dropout(hparams.dropout)
        self.use_positional_encoding = hparams.use_positional_encoding

        if self.use_positional_encoding:
            self.positional_encoder = PositionEncoder(
                hparams.embedding_dim, hparams.title_size, hparams.dropout)

        if hparams.use_time_embedding:
            self.use_time_embedding = True
            self.time_embedding = nn.Linear(hparams.time_size, hparams.time_embedding_dim)
            self.self_attention = SelfAttention(hparams.embedding_dim + hparams.time_embedding_dim, hparams.head_num, hparams.head_dim)
        else:
            self.use_time_embedding = False
            self.self_attention = SelfAttention(hparams.embedding_dim, hparams.head_num, hparams.head_dim)
        
        self.dense_layers = nn.Sequential(
            nn.Linear(hparams.head_num * hparams.head_dim,
                      hparams.linear_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hparams.linear_hidden_dim),
            nn.Dropout(hparams.dropout),
            nn.Linear(hparams.linear_hidden_dim, hparams.linear_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hparams.linear_hidden_dim),
            nn.Dropout(hparams.dropout),
            nn.Linear(hparams.linear_hidden_dim,
                      hparams.head_num * hparams.head_dim),
            nn.ReLU(),
            nn.LayerNorm(hparams.head_num * hparams.head_dim),
            nn.Dropout(hparams.dropout),
        )

        self.att_layer = AdditiveAttention(
            hparams.head_num * hparams.head_dim, hparams.attention_hidden_dim)

    def forward(self, sequences_input_title, timestamps=None):
        sequences_input_title = sequences_input_title.long()        # Shape: (title_size)
        embedded_sequences = self.embedding(sequences_input_title)  # Shape: (title_size, embedding_dim)
        if self.use_positional_encoding:
            embedded_sequences = self.positional_encoder(embedded_sequences) # Shape: (title_size, embedding_dim)
        if self.use_time_embedding:
            timestamps = timestamps.long() # Shape: (title_size)
            time_embedded = self.time_embedding(timestamps) # Shape: (title_size, time_embedding_dim)
            embedded_sequences = torch.cat([embedded_sequences, time_embedded], dim=-1) # Shape: (title_size, embedding_dim + time_embedding_dim)
        y = self.dropout(embedded_sequences) # Shape: (title_size, embedding_dim)
        y = self.self_attention(y, y, y)  # Shape: (title_size, head_num * head_dim)
        y = self.dense_layers(y)  # Shape: (title_size, head_num * head_dim)
        y = self.att_layer(y) # Shape: (head_num * head_dim)
        return y


class UserEncoder(nn.Module):
    def __init__(self, hparams, news_encoder):
        super(UserEncoder, self).__init__()
        self.news_encoder = news_encoder
        self.self_attention = SelfAttention(
            hparams.head_num * hparams.head_dim, hparams.head_num, hparams.head_dim)
        self.att_layer = AdditiveAttention(
            hparams.head_num * hparams.head_dim, hparams.attention_hidden_dim)
        
        self.head_num = hparams.head_num   
        self.head_dim = hparams.head_dim

    def forward(self, history, timestamps=None): # Shape: (history_size, title_size)
        history_representations = torch.zeros(history.shape[0], self.head_num * self.head_dim).to(history.device) 
        for i in range(history.shape[0]):  # for each news in history encode the news
            if timestamps is not None: 
                history_representations[i] = self.news_encoder(history[i], timestamps[i])  # Shape: (head_num * head_dim)
            else:
                history_representations[i] = self.news_encoder(history[i]) # Shape: (head_num * head_dim)
        y = self.self_attention(history_representations,
                                history_representations, history_representations) # Shape: (history_size, head_num * head_dim)
        y = self.att_layer(y) # Shape: (head_num * head_dim)
        return y 


class ClickPredictor(nn.Module):
    def __init__(self, hparams):
        super().__init__()

    def forward(self, news_representation, user_representation): 
        prob = torch.einsum('cd,d->c', news_representation, user_representation)  # Shape: (candidate_size)
        return prob


class NRMSModel(nn.Module):
    def __init__(self, hparams, word2vec_embedding):
        super().__init__()

        self.news_encoder = NewsEncoder(hparams, word2vec_embedding)
        self.user_encoder = UserEncoder(hparams, self.news_encoder)
        self.click_predictor = ClickPredictor(hparams)

        self.head_num = hparams.head_num   
        self.head_dim = hparams.head_dim

    # History shape: (history_size, title_size)
    # Candidates shape: (candidate_size, title_size)
    def forward(self, candidates, history, candidate_timestamps = None, history_timestamps=None):           
        user_representation = self.user_encoder(history, history_timestamps)  # u in the paper
        
        news_representations = torch.zeros(candidates.shape[0], self.head_num * self.head_dim).to(candidates.device) # candidate_size, head_num * head_dim
        for i in range(candidates.shape[0]): # for each candidate news encode the news
            if candidate_timestamps is not None:
                news_representations[i] = self.news_encoder(candidates[i], candidate_timestamps[i])
            else:
                news_representations[i] = self.news_encoder(candidates[i])
                
        click_probability = self.click_predictor(news_representations, user_representation)
        
        p_i = F.softmax(click_probability, dim=-1)

        return p_i
