# ==========================================================
# IMPORT LIBRARY
# ==========================================================

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Embedding,
    SimpleRNN,
    LSTM,
    GRU,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import EarlyStopping
# ==========================================================
# CONFIGURATION
# ==========================================================

DATASET_PATH = "dataset/IMDB Dataset.csv"

OUTPUT_DIR = "output"

MODEL_DIR = "models"

VOCAB_SIZE = 10000

MAX_LENGTH = 200

EMBEDDING_DIM = 128

TEST_SIZE = 0.2

RANDOM_STATE = 42

BATCH_SIZE = 64

EPOCHS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 60)
print("LOAD DATASET")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

print(df.head())

print()

print("Jumlah Data :", len(df))

print()

print(df["sentiment"].value_counts())

# ==========================================================
# PREPROCESSING
# ==========================================================

print("\nCleaning text...")

def clean_text(text):

    text = text.lower()

    text = re.sub("<br />", " ", text)

    text = re.sub(r"[^a-zA-Z ]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

df["review"] = df["review"].apply(clean_text)

print("Cleaning selesai.")

# ==========================================================
# LABEL ENCODING
# ==========================================================

encoder = LabelEncoder()

df["sentiment"] = encoder.fit_transform(df["sentiment"])

print()

print(df.head())

# ==========================================================
# TOKENIZER
# ==========================================================

print("\nTokenizer...")

tokenizer = Tokenizer(

    num_words=VOCAB_SIZE,

    oov_token="<OOV>"

)

tokenizer.fit_on_texts(df["review"])

print("Vocabulary Size :", len(tokenizer.word_index))

# ==========================================================
# TEXT TO SEQUENCE
# ==========================================================

sequences = tokenizer.texts_to_sequences(

    df["review"]

)

# ==========================================================
# PADDING
# ==========================================================

X = pad_sequences(

    sequences,

    maxlen=MAX_LENGTH,

    padding="post",

    truncating="post"

)

y = df["sentiment"]

print()

print("Shape X :", X.shape)

print("Shape y :", y.shape)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y

)

print()

print("Train :", X_train.shape)

print("Test  :", X_test.shape)

# ==========================================================
# EARLY STOPPING
# ==========================================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=2,

    restore_best_weights=True

)
# ==========================================================
# FUNCTION : BUILD MODEL
# ==========================================================

def build_rnn():

    model = Sequential([

        Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=EMBEDDING_DIM,
            input_length=MAX_LENGTH
        ),

        SimpleRNN(64),

        Dropout(0.5),

        Dense(32, activation="relu"),

        Dense(1, activation="sigmoid")

    ])

    model.compile(

        optimizer="adam",

        loss="binary_crossentropy",

        metrics=["accuracy"]

    )

    return model


# ----------------------------------------------------------


def build_lstm():

    model = Sequential([

        Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=EMBEDDING_DIM,
            input_length=MAX_LENGTH
        ),

        LSTM(64),

        Dropout(0.5),

        Dense(32, activation="relu"),

        Dense(1, activation="sigmoid")

    ])

    model.compile(

        optimizer="adam",

        loss="binary_crossentropy",

        metrics=["accuracy"]

    )

    return model


# ----------------------------------------------------------


def build_gru():

    model = Sequential([

        Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=EMBEDDING_DIM,
            input_length=MAX_LENGTH
        ),

        GRU(64),

        Dropout(0.5),

        Dense(32, activation="relu"),

        Dense(1, activation="sigmoid")

    ])

    model.compile(

        optimizer="adam",

        loss="binary_crossentropy",

        metrics=["accuracy"]

    )

    return model


# ==========================================================
# CREATE MODEL
# ==========================================================

print("=" * 60)
print("MEMBUAT MODEL")
print("=" * 60)

rnn_model = build_rnn()

lstm_model = build_lstm()

gru_model = build_gru()

print("\nRNN")
rnn_model.summary()

print("\nLSTM")
lstm_model.summary()

print("\nGRU")
gru_model.summary()
# ==========================================================
# TRAIN RNN
# ==========================================================

print("\n" + "="*60)
print("TRAINING RNN")
print("="*60)

history_rnn = rnn_model.fit(

    X_train,
    y_train,

    validation_split=0.2,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[early_stop],

    verbose=1

)

rnn_model.save(

    os.path.join(MODEL_DIR, "rnn.keras")

)

print("Model RNN berhasil disimpan")


# ==========================================================
# TRAIN LSTM
# ==========================================================

print("\n" + "="*60)
print("TRAINING LSTM")
print("="*60)

history_lstm = lstm_model.fit(

    X_train,
    y_train,

    validation_split=0.2,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[early_stop],

    verbose=1

)

lstm_model.save(

    os.path.join(MODEL_DIR, "lstm.keras")

)

print("Model LSTM berhasil disimpan")


# ==========================================================
# TRAIN GRU
# ==========================================================

print("\n" + "="*60)
print("TRAINING GRU")
print("="*60)

history_gru = gru_model.fit(

    X_train,
    y_train,

    validation_split=0.2,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[early_stop],

    verbose=1

)

gru_model.save(

    os.path.join(MODEL_DIR, "gru.keras")

)

print("Model GRU berhasil disimpan")

# ==========================================================
# FUNCTION : EVALUATION
# ==========================================================

def evaluate_model(model, X_test, y_test, model_name):

    print("\n" + "=" * 60)
    print(f"EVALUASI MODEL : {model_name}")
    print("=" * 60)

    # Prediksi probabilitas
    y_prob = model.predict(X_test, verbose=0)

    # Konversi menjadi 0 / 1
    y_pred = (y_prob > 0.5).astype(int)

    # Hitung metrik
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {pre:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nClassification Report")

    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(f"Confusion Matrix - {model_name}")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            OUTPUT_DIR,
            f"{model_name.lower()}_confusion_matrix.png"
        ),
        dpi=300

    )

    plt.close()

    return {

        "Model": model_name,

        "Accuracy": acc,

        "Precision": pre,

        "Recall": rec,

        "F1 Score": f1

    }


# ==========================================================
# EVALUATION
# ==========================================================

result_rnn = evaluate_model(

    rnn_model,

    X_test,

    y_test,

    "RNN"

)

result_lstm = evaluate_model(

    lstm_model,

    X_test,

    y_test,

    "LSTM"

)

result_gru = evaluate_model(

    gru_model,

    X_test,

    y_test,

    "GRU"

)