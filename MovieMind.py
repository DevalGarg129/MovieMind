import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout,
    Bidirectional
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

VOCAB_SIZE = 10000
MAX_LENGTH = 250

(X_train_raw, y_train), (X_test_raw, y_test) = imdb.load_data(
    num_words=VOCAB_SIZE
)
print("Training Reviews : ", len(X_train_raw))
print("Testing Reviews : ", len(X_test_raw))
print("First Review Label : ", y_train[0])
print("First Review Token IDs: ", X_train_raw[0][:20])

X_train = pad_sequences(
    X_train_raw,
    maxlen=MAX_LENGTH,
    padding='post',
    truncating='post'
)

X_test = pad_sequences(
    X_test_raw,
    maxlen=MAX_LENGTH,
    padding='post',
    truncating='post'
)

print("Training shape : ", X_train.shape)
print("Testing shape : ", X_test.shape)
print("Training labels:", np.unique(y_train))

tf.keras.utils.set_random_seed(42)
model = Sequential([
    tf.keras.layers.Input(shape=(MAX_LENGTH,)),
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=128,
        mask_zero=True
    ),

    Bidirectional(LSTM(32)),
    Dropout(0.4),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=['accuracy']
)

model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=1,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=128,
    callbacks=[early_stopping],
    verbose=1
)

test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=128, verbose=1)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

test_probabilities = model.predict(X_test, batch_size=128, verbose=1).ravel()

y_pred = (test_probabilities >= 0.5).astype(int)

print("Probability range:",test_probabilities.min(),test_probabilities.max())
print("Prediction standard deviation:", test_probabilities.std())
print("Training accuracy recorded in history:", history.history["accuracy"][-1])

VOCAB_SIZE = 10000
MAX_LENGTH = 250

word_index = imdb.get_word_index()

# Reserve the special indices used by the IMDb dataset.
word_to_id = {
    word: index + 3
    for word, index in word_index.items()
    if index + 3 < VOCAB_SIZE
}

word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
word_to_id["<UNUSED>"] = 3

print("Vocabulary size:", len(word_to_id))

def preprocess_review(review):
    words = review.lower().split()

    sequence = [1]  # IMDb start token

    for word in words:
        # Basic tokenization for our first version.
        # IMDb uses word tokens, so remove surrounding punctuation.
        word = word.strip(".,!?;:\"'()[]{}")

        if word:
            sequence.append(word_to_id.get(word, 2))

    padded = pad_sequences(
        [sequence],
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post",
        value=0
    )

    return padded


def predict_sentiment(review):
    processed = preprocess_review(review)

    probability = float(
        model.predict(processed, verbose=0)[0][0]
    )

    sentiment = "Positive" if probability >= 0.5 else "Negative"

    return {
        "review": review,
        "sentiment": sentiment,
        "positive_probability": round(probability, 4),
        "negative_probability": round(1 - probability, 4)
    }

reviews = [
    "This movie was absolutely fantastic and entertaining.",
    "The film was boring, predictable, and disappointing.",
    "The acting was brilliant but the story was terrible.",
    "I loved every minute of this wonderful movie."
]

for review in reviews:
    print(predict_sentiment(review))
    print("-" * 60)

y_pred = (test_probabilities >= 0.5).astype(int)

print(classification_report(
    y_test,
    y_pred,
    target_names=["Negative", "Positive"],
    digits=4
))

print("ROC-AUC:", round(
    roc_auc_score(y_test, test_probabilities), 4
))

cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"]
).plot(values_format="d")

plt.title("MovieMind — Confusion Matrix")
plt.show()