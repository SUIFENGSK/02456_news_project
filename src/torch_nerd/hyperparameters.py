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
    # MODEL ARCHITECTURE
    head_num: int = 20
    head_dim: int = 20
    attention_hidden_dim: int = 200
    linear_hidden_dim: int = 80

    # POSITIONAL ENCODER
    use_positional_encoding: bool = False
    use_learned_positions: bool = False

    # MODEL OPTIMIZER:
    optimizer: str = "adam"
    loss: str = "cross_entropy_loss"
    dropout: float = 0.2
    learning_rate: float = 1e-4
