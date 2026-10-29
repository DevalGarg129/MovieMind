# MovieMind

MovieMind is a small movie-review sentiment classifier built with TensorFlow/Keras. It trains a bidirectional LSTM on the IMDb reviews dataset and classifies review text as **Positive** or **Negative**.

## Features

- Loads the IMDb dataset through `tensorflow.keras.datasets.imdb`.
- Trains a neural network using the 10,000 most frequent tokens and review sequences padded/truncated to 250 tokens.
- Uses an embedding layer, a bidirectional LSTM, dropout, and dense layers for binary sentiment classification.
- Evaluates the model on the IMDb test split and prints classification metrics.
- Runs a few sample predictions, including the predicted sentiment and positive/negative probabilities.
- Displays a confusion-matrix plot.

## Requirements

- Python
- The packages listed in [requirements.txt](requirements.txt)

Use a Python version supported by the installed TensorFlow release.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\\Scripts\\activate` instead of the `source` command.

## Run

```bash
python MovieMind.py
```

The first run may need an internet connection to download the IMDb dataset. The script trains the model each time it runs; it does not currently save or load a trained model. Training progress, evaluation results, sample predictions, and a confusion matrix are printed or displayed in the terminal/plot window.

## Model configuration

The main settings are defined near the top of `MovieMind.py`:

- `VOCAB_SIZE = 10000`: maximum token ID vocabulary size.
- `MAX_LENGTH = 250`: padded/truncated review length.
- Training uses a 20% validation split, a batch size of 128, and up to 10 epochs, with early stopping on validation loss.

To try a review with the trained model, pass its text to `predict_sentiment(review)` in the script. The function returns the original review, a sentiment label, and the estimated positive and negative probabilities.

## Current limitation

The script calls `roc_auc_score` near the end but does not currently import it. As written, it will raise a `NameError` at that metric unless `roc_auc_score` is added to the `sklearn.metrics` import in `MovieMind.py`.
