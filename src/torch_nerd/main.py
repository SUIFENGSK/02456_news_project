from from_ebrec.evaluation import AucScore, MrrScore, NdcgScore
from from_ebrec.evaluation import MetricEvaluator
import matplotlib.pyplot as plt
from tqdm import tqdm  # for progress bars
import torch.optim as optim
import torch.nn as nn
from hyperparameters import hparams_nrms
from nrms import NRMSModel
from src.torch_nerd.dataset import NRMSDataLoader
from from_ebrec._articles import create_article_id_to_value_mapping
from from_ebrec._articles import convert_text2encoding_with_transformers
from from_ebrec._polars import concat_str_columns
from from_ebrec._nlp import get_transformers_word_embeddings
import transformers as huggingface
import from_ebrec._constants as cs
from utils.data_handler import NewsDataset
import torch
from pathlib import Path
import sys
import os
BLACKHOLE = False


if BLACKHOLE:
    workspace_path = os.path.expandvars('$BLACKHOLE')
    sys.path.append(workspace_path+'/DeepLearning/02456_news_project/src')
    DATAPATH = Path(workspace_path+"/DeepLearning/ebnerd_data").expanduser()
else:
    DATAPATH = Path("~/ebnerd_data").expanduser()

DATASET = "ebnerd_demo"
# DATASET = "ebnerd_small"


print("torch version:", torch.__version__)

# Check gpu availability


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Test:
# print(torch.zeros(1).cuda())

SEED = 42
HISTORY_SIZE = 50

COLS = [
    cs.DEFAULT_USER_COL,
    cs.DEFAULT_IMPRESSION_ID_COL,
    cs.DEFAULT_IMPRESSION_TIMESTAMP_COL,
    cs.DEFAULT_HISTORY_ARTICLE_ID_COL,
    cs.DEFAULT_CLICKED_ARTICLES_COL,
    cs.DEFAULT_INVIEW_ARTICLES_COL,

    cs.DEFAULT_ARTICLE_ID_COL,
    cs.DEFAULT_AGE_COL,
    cs.DEFAULT_READ_TIME_COL
]

FRACTION = 0.01
# FRACTION = 0.1
# FRACTION = 1

# test
dataset = NewsDataset()

dataset.setup_df(dataset_path=DATAPATH, datasplit=DATASET,
                 history_size=HISTORY_SIZE, columns=COLS, fraction=FRACTION, seed=SEED)


dataset.setup_articles_data(dataset_path=DATAPATH.joinpath(DATASET))

df_articles = dataset.df_articles

TRANSFORMER_MODEL_NAME = "FacebookAI/xlm-roberta-base"
TEXT_COLUMNS_TO_USE = [cs.DEFAULT_SUBTITLE_COL, cs.DEFAULT_TITLE_COL]
MAX_TITLE_LENGTH = 30

# LOAD HUGGINGFACE:
transformer_model = huggingface.AutoModel.from_pretrained(
    TRANSFORMER_MODEL_NAME)
transformer_tokenizer = huggingface.AutoTokenizer.from_pretrained(
    TRANSFORMER_MODEL_NAME)

word2vec_embedding = get_transformers_word_embeddings(transformer_model)
df_articles, cat_cal = concat_str_columns(
    df_articles, columns=TEXT_COLUMNS_TO_USE)
df_articles, token_col_title = convert_text2encoding_with_transformers(
    df_articles, transformer_tokenizer, cat_cal, max_length=MAX_TITLE_LENGTH)
article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col=token_col_title)


BATCH_SIZE = 64

train_dataloader = NRMSDataLoader(
    behaviors=dataset.df_train,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=cs.DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,
    batch_size=BATCH_SIZE,
)
val_dataloader = NRMSDataLoader(
    behaviors=dataset.df_validation,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=cs.DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,
    batch_size=BATCH_SIZE,
)


hparams = hparams_nrms()

# PARAMETERS
hparams.title_size = MAX_TITLE_LENGTH
hparams.history_size = HISTORY_SIZE

# MODEL ARCHITECTURE
hparams.head_num = 20
hparams.head_dim = 20
hparams.attention_hidden_dim = 200
hparams.linear_hidden_dim = 400

hparams.use_positional_encoding = True
hparams.use_learned_positions = True

# MODEL OPTIMIZER:
hparams.optimizer = "adam"
hparams.loss = "cross_entropy_loss"
hparams.dropout = 0.2
hparams.learning_rate = 1e-4

model = NRMSModel(
    hparams=hparams, word2vec_embedding=word2vec_embedding, debug=False)

print(model)


# Define the loss function and optimizer
if hparams.loss == "cross_entropy_loss":
    criterion = nn.CrossEntropyLoss()
elif hparams.loss == "mse_loss":
    criterion = nn.MSELoss()
else:
    raise ValueError(f"Loss function {hparams.loss} not supported")

if hparams.optimizer == "adam":
    optimizer = optim.Adam(model.parameters(), lr=hparams_nrms.learning_rate)
else:
    raise ValueError(f"Optimizer {hparams.optimizer} not supported")
# Train the model

EPOCHS = 10

# Move model to GPU if available
model.to(device)

# Training loop
train_loss_history, val_loss_history = [], []

for epoch in range(EPOCHS):
    model.train()  # Set the model to training mode
    train_loss = 0.0

    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}"):
        # Unpacking of batch
        (inputs, labels) = batch
        his_input_title, pred_input_title = inputs

        # Move data to device
        his_input_title = his_input_title.to(device)
        pred_input_title = pred_input_title.to(device)

        labels = labels.to(device)

        # Forward pass
        outputs = model(pred_input_title, his_input_title)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}/{EPOCHS}"):
            (inputs, labels) = batch
            his_input_title, pred_input_title = inputs

            his_input_title = his_input_title.to(device)
            pred_input_title = pred_input_title.to(device)
            labels = labels.to(device)

            outputs = model(pred_input_title, his_input_title)
            loss = criterion(outputs, labels)

            # print("Output")
            # print(outputs.tolist())
            # print("Labels")
            # print(labels.tolist())
            # print("")

            val_loss += loss.item()

    train_loss /= len(train_dataloader)
    val_loss /= len(val_dataloader)
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

# Plot the loss history

plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Evaluate the model
BATCH_SIZE_TEST = 1  # Currently onoy supports batch size 1

dataset.setup_test_data(dataset_path=DATAPATH, datasplit=DATASET,
                        history_size=HISTORY_SIZE, columns=COLS, fraction=FRACTION, seed=SEED)

test_dataloader = NRMSDataLoader(
    behaviors=dataset.df_test,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=cs.DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,  # Is true in EBREC, but then it does not work
    batch_size=BATCH_SIZE_TEST,
)

# go from [[a], [b], [c]] to [a, b, c]


def convert_to_list(l):
    return [(item) for sublist in l for item in sublist]


model.eval()

pred_test = []
with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Test"):
        (inputs, labels) = batch
        his_input_title, pred_input_title = inputs

        his_input_title = his_input_title.to(device)
        pred_input_title = pred_input_title.to(device)
        labels = labels.to(device)

        outputs = model(pred_input_title, his_input_title)

        # print("Output")
        # print(outputs.tolist())
        # print("Labels")
        # print(labels.tolist())
        bob = convert_to_list(outputs.tolist())
        # print(bob)
        # print(convert_to_int_list(labels.tolist()))
        # print("")

        pred_test.append(bob)

metrics = MetricEvaluator(
    labels=dataset.df_test["labels"].to_list(),
    predictions=pred_test,
    metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
)
metrics.evaluate()
number_to_print = 20
print("Top %d predictions vs labels:" % number_to_print)
labels = dataset.df_test["labels"].to_list()
for i in range(number_to_print):
    print(f"Article {i}")
    for j in range(len(pred_test[i])):
        print(f"{pred_test[i][j]:.3f} vs {labels[i][j]:.3f}")
    print("")
