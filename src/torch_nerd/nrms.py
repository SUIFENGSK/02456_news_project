import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import AttLayer2 as AdditiveAttention
from layers import SelfAttention as SelfAttention
from layers import PositionEncoder as PositionEncoder


class NewsEncoder(nn.Module):
    def __init__(self, hparams, word2vec_embedding, debug=False):
        super(NewsEncoder, self).__init__()
        self.embedding = word2vec_embedding
        self.dropout = nn.Dropout(hparams.dropout)
        self.use_positional_encoding = hparams.use_positional_encoding

        if self.use_positional_encoding:
            self.positional_encoder = PositionEncoder(
                hparams.embedding_dim, hparams.title_size, hparams.dropout)

        self.self_attention = SelfAttention(
            hparams.embedding_dim, hparams.head_num, hparams.head_dim
            )
        
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

        self.debug = debug

        self.batch_size = hparams.batch_size
        self.title_size = hparams.title_size
        self.head_num = hparams.head_num
        self.head_dim = hparams.head_dim
        self.embedding_dim = hparams.embedding_dim

    def forward(self, sequences_input_title):
        # Convert from (batch_size, history_size, title_size) to (batch_size, title_size)
        history_size_saved = False
        if sequences_input_title.dim() == 3:
            history_size_saved = True
            history_size = sequences_input_title.shape[1]
            sequences_input_title = sequences_input_title[:, -1, :]

        if self.debug:
            print("NE0: Shape of input:", sequences_input_title.shape,
                  ". Should be (batch_size, title_size): (" + str(self.batch_size) + ", " + str(self.title_size) + ")")
        sequences_input_title = sequences_input_title.long()

        if self.debug:
            print("NE2: Shape after casting to long:",
                  sequences_input_title.shape, ". Should be (batch_size, title_size): (" + str(self.batch_size) + ", " + str(self.title_size) + ")")
        embedded_sequences = self.embedding(sequences_input_title)

        if self.use_positional_encoding:
            embedded_sequences = self.positional_encoder(embedded_sequences)

        if self.debug:
            print("NE3: Shape after embedding:", embedded_sequences.shape,
                  ". Should be (batch_size, title_size, embedding_dim): (" + str(self.batch_size) + ", " + str(self.title_size) + ", " + str(self.embedding_dim) + ")")
        y = self.dropout(embedded_sequences)

        if self.debug:
            print("NE4: Shape after dropout:", y.shape,
                  ". Should be (batch_size, title_size, embedding_dim): (" + str(self.batch_size) + ", " + str(self.title_size) + ", " + str(self.embedding_dim) + ")")
        y = self.self_attention(y, y, y)

        if self.debug:
            print("NE5: Shape after self attention:", y.shape,
                  ". Should be (batch_size, title_size, head_num * head_dim): (" + str(self.batch_size) + ", " + str(self.title_size) + ", " + str(self.head_num * self.head_dim) + ")")
        #y = self.dense_layers(y)

        if self.debug:
            print("NE6: Shape after dense layers:", y.shape,
                  ". Should be (batch_size, title_size, head_num * head_dim): (" + str(self.batch_size) + ", " + str(self.title_size) + ", " + str(self.head_num * self.head_dim) + ")")
        y = self.att_layer(y)

        if self.debug:
            print("NE7: Shape after att layer:", y.shape,
                  ". Should be (batch_size, head_num * head_dim): (" + str(self.batch_size) + ", " + str(self.head_num * self.head_dim) + ")")

        # Re add the history size dimension
        if history_size_saved:
            y = y.unsqueeze(1).repeat(1, history_size, 1)
        return y


class UserEncoder(nn.Module):
    def __init__(self, hparams, title_encoder, debug=False):
        super(UserEncoder, self).__init__()
        self.title_encoder = title_encoder
        self.self_attention = SelfAttention(
            hparams.head_num * hparams.head_dim, hparams.head_num, hparams.head_dim)
        self.att_layer = AdditiveAttention(
            hparams.head_num * hparams.head_dim, hparams.attention_hidden_dim)

        self.debug = debug
        self.batch_size = hparams.batch_size
        self.history_size = hparams.history_size
        self.title_size = hparams.title_size
        self.head_num = hparams.head_num
        self.head_dim = hparams.head_dim

    def forward(self, his_input_title):
        if self.debug:
            print("UE1: Shape of input:", his_input_title.shape,
                  ". Should be (batch_size, history_size, title_size): (" + str(self.batch_size) + ", " + str(self.history_size) + ", " + str(self.title_size) + ")")
        click_title_presents = self.title_encoder(his_input_title)

        if self.debug:
            print("UE2: Shape after title encoder:", click_title_presents.shape, ". Should be (batch_size, history_size, head_num * head_dim): (" + str(self.batch_size) + ", " + str(self.history_size) + ", " + str(self.head_num * self.head_dim) + ")")
        y = self.self_attention(click_title_presents,
                                click_title_presents, click_title_presents)

        if self.debug:
            print("UE3: Shape after self attention ", y.shape,
                  ". Should be (batch_size, history_size, head_num * head_dim): (" + str(self.batch_size) + ", " + str(self.history_size) + ", " + str(self.head_num * self.head_dim) + ")")
        y = self.att_layer(y)

        if self.debug:
            print("UE4: Shape after att layer ", y.shape,
                  ". Should be (batch_size, head_num * head_dim): (" + str(self.batch_size) + ", " + str(self.head_num * self.head_dim) + ")")
        return y


class ClickPredictor(nn.Module):
    def __init__(self, hparams, debug=False):
        super().__init__()
        self.debug = debug
        self.batch_size = hparams.batch_size
        self.candidate_size = hparams.candidate_size
        self.head_num = hparams.head_num
        self.head_dim = hparams.head_dim

    def forward(self, news_representation, user_representation):
        if self.debug:
            print("CP1: Shape of news_representation:", news_representation.shape,
                  ". Should be (batch_size, candidate_size, head_num * head_dim): (" + str(self.batch_size) + ", " + str(self.candidate_size) + ", " + str(self.head_num * self.head_dim) + ")")
            print("CP2: Shape of user_representation:", user_representation.shape,
                  ". Should be (batch_size, head_num * head_dim): (" + str(self.batch_size) + ", " + str(self.head_num * self.head_dim) + ")")

        # Reshape the news representation to (batch_size, candidate_size, head_num * head_dim)
        news_representation = news_representation.permute(1, 0, 2)
        if self.debug:
            print("CP3: Reshape of news_representation:", news_representation.shape,
                  ". Should be (candidate_size, batch_size, head_num * head_dim): (" + str(self.candidate_size) + ", " + str(self.batch_size) + ", " + str(self.head_num * self.head_dim) + ")")
            
        # Compute the dot product between the news and user representations so that the output is (batch_size, candidate_size)
        prob = torch.bmm(news_representation,
                         user_representation.unsqueeze(2)).squeeze(2)
        if self.debug:
            print("CP3: Shape of prob:", prob.shape,
                  ". Should be (batch_size, candidate_size): (" + str(self.batch_size) + ", " + str(self.candidate_size) + ")")
        return prob


class NRMSModel(nn.Module):
    def __init__(self, hparams, word2vec_embedding, debug=False):
        super().__init__()
        tensor_word2vec_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(word2vec_embedding), freeze=False)

        self.news_encoder = NewsEncoder(hparams, tensor_word2vec_embedding, debug)
        self.user_encoder = UserEncoder(hparams, self.news_encoder, debug)
        self.click_predictor = ClickPredictor(hparams, debug)

        self.debug = debug

        self.batch_size = hparams.batch_size
        self.candidate_size = hparams.candidate_size
        self.history_size = hparams.history_size
        self.title_size = hparams.title_size
        self.head_num = hparams.head_num   
        self.head_dim = hparams.head_dim


    def forward(self, pred_input_title, his_input_title):
        if self.debug:
            print("Model: Shape of pred_input_title:", pred_input_title.shape,
                  ". Should be (batch_size, candidate_size, title_size): (" + str(self.batch_size) + ", " + str(self.candidate_size) + ", " + str(self.title_size) + ")")
            print("Model: Shape of his_input_title:", his_input_title.shape,
                  ". Should be (batch_size, history_size, title_size): (" + str(self.batch_size) + ", " + str(self.history_size) + ", " + str(self.title_size) + ")")
            
        user_representation = self.user_encoder(his_input_title)  # u in the paper
        
        # Reshape the candidate news to (candidate_size,  batch_size, title_size)
        pred_input_title = pred_input_title.permute(1, 0, 2)


        news_representations = torch.zeros(pred_input_title.shape[0], pred_input_title.shape[1], self.head_num * self.head_dim)
        news_representations = news_representations.to(pred_input_title.device)
        for i in range(pred_input_title.shape[0]):
            news_representations[i] = self.news_encoder(pred_input_title[i])
        

        if self.debug:
            print("Model: Shape of user_representation:", user_representation.shape,
                  ". Should be (batch_size, head_num * head_dim): (" + str(self.batch_size) + ", " + str(self.head_num * self.head_dim) + ")")
            print("Model: Shape of news_representations:", news_representations.shape,
                  ". Should be (candidate_size, batch_size, head_num * head_dim): (" + str(self.candidate_size) + ", " + str(self.batch_size) + ", " + str(self.head_num * self.head_dim) + ")")
        click_probability = self.click_predictor(
            news_representations, user_representation)  # y_hat in the paper
        p_i = F.softmax(click_probability, dim=-1)
        return p_i
