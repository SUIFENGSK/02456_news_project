class HyperParameters:
    def __init__(self, vocab_size, word_embedding_dim, dropout, head_num, head_dim, attention_hidden_dim, title_size):
        self.vocab_size = vocab_size # vocab size -> how many words in the vocab dictionary (input size)
        self.word_embedding_dim = word_embedding_dim # word embedding dimension -> how many dimensions for each word (output size)
        self.dropout = dropout # dropout rate -> how many neurons to drop out during training to prevent overfitting
        self.head_num = head_num # number of heads in the multi-head attention model -> lower number of heads means more global attention, higher number of heads means more local attention
        self.head_dim = head_dim # dimension of each head in the multi-head attention -> lower number of head_dim means more global attention, higher number of head_dim means more local attention
        self.attention_hidden_dim = attention_hidden_dim # hidden dimension in the feedforward network in the transformer model -> higher number of attention_hidden_dim means more complex model
        self.title_size = title_size # title size -> how many words in the title of the news article (input size)

