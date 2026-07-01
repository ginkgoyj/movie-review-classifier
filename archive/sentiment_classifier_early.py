# =============================================================================
# Movie Review Classification
# Dataset: NLTK movie_reviews corpus
# Models: Multinomial Naive Bayes, Linear SVM
# =============================================================================

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk.corpus import movie_reviews
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR
NLTK_DATA_DIR = BASE_DIR / "nltk_data"
NLTK_DATA_DIR.mkdir(exist_ok=True)
nltk.data.path.insert(0, str(NLTK_DATA_DIR))

stemmer = PorterStemmer()


def ensure_nltk_resource(resource_path, package_name):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        downloaded = nltk.download(package_name, download_dir=str(NLTK_DATA_DIR), quiet=True)
        if not downloaded:
            raise RuntimeError(f"Failed to download NLTK resource: {package_name}")


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    tokens = re.findall(r"[a-z]+", text)
    tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in ENGLISH_STOP_WORDS and len(token) > 2
    ]
    return " ".join(tokens)


def save_model_comparison(nb_scores, svm_scores):
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, nb_scores, width, label="Naive Bayes", color="steelblue")
    bars2 = ax.bar(x + width / 2, svm_scores, width, label="Linear SVM", color="coral")

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Naive Bayes vs Linear SVM")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.7, 1.0)
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=8)
    plt.tight_layout()
    output_path = OUTPUT_DIR / "model_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nSaved: {output_path}")


def save_confusion_matrices(y_test, nb_preds, svm_preds):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, preds, title in zip(
        axes,
        [nb_preds, svm_preds],
        ["Naive Bayes", "Linear SVM"],
    ):
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(title)