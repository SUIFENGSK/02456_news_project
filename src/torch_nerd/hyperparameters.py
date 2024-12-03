# From newsrec folder:
DEFAULT_TITLE_SIZE = 30
DEFAULT_BODY_SIZE = 40
UNKNOWN_TITLE_VALUE = [0] * DEFAULT_TITLE_SIZE
UNKNOWN_BODY_VALUE = [0] * DEFAULT_BODY_SIZE

DEFAULT_DOCUMENT_SIZE = 768

class hparams_nrms:
    # INPUT DIMENTIONS:
    title_size: int = DEFAULT_TITLE_SIZE
    history_size: int = 20
    batch_size: int = 32
    candidate_size: int = 200
    embedding_dim: int = 300


    # MODEL ARCHITECTURE
    head_num: int = 20
    head_dim: int = 20
    attention_hidden_dim: int = 200
    linear_hidden_dim: int = 80

    # TIME EMBEDDING
    use_time_embedding: bool = False
    time_size: int = 100
    time_embedding_dim: int = 100

    # POSITIONAL ENCODER
    use_positional_encoding: bool = False

    # MODEL OPTIMIZER:
    optimizer: str = "adam"
    loss: str = "cross_entropy_loss"
    dropout: float = 0.2
    learning_rate: float = 1e-4
