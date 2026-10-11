import os
import re

import matplotlib
matplotlib.use("Agg")  # save figures to disk without opening a window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold

DATASET_DIR = "dataset_landmarks"
RESULTS_DIR = "results"

N_FOLDS = 5  # number of folds for cross-validation
GAP = 10  # samples removed from training on each side of a test block (purging)
N_ESTIMATORS = 500  # same as training.py, so we evaluate the same kind of model
SEED = 42  # fixed seed so results are reproducible
LAGS = [1, 2, 5, 10, 20, 50, 100]  # distances (in samples) used in the redundancy analysis


def normalize(sample):
    """
    Transform the array into a 21:3 matrix and normalize it using the wrist as the origin. This is with the objective
    of maintaining consistency in the data while training the model and to avoid that the position of the hand
    affects the prediction
    :param
    sample: np.ndarray
    Array of size 63 with the (x,y,z) coordinates of the 21 mediapipe landmarks in the hand
    :return:
    np.ndarray
    Array of size 63 with the normalized coordinates of the 21 mediapipe landmarks in the hand
    """
    pts = sample.reshape(21, 3)
    pts = pts - pts[0]

    max_val = np.max(np.abs(pts))
    if max_val != 0:
        pts = pts / max_val

    return pts.flatten()


def natural_key(name):
    """
    Builds a sort key that orders file names the way a human would ("2.npy" before "10.npy"), so the samples are
    loaded in the same order they were recorded
    :param
    name: str
    File name
    :return:
    list
    Mix of strings and integers used as the sort key
    """
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def load_dataset():
    """
    Loads every sample of the dataset, normalizes it and keeps track of its position inside its letter. The position
    is the recording order, which is what we use to build time-based (blocked) splits
    :return:
    tuple(np.ndarray, np.ndarray, np.ndarray)
    X with the normalized samples, y with the letters and positions with the recording order inside each letter
    """
    X, y, positions = [], [], []

    for label in sorted(os.listdir(DATASET_DIR)):
        class_dir = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(class_dir):
            continue

        files = sorted((f for f in os.listdir(class_dir) if f.endswith(".npy")), key=natural_key)

        position = 0
        for file in files:
            data = np.load(os.path.join(class_dir, file))
            if data.ndim == 1:
                data = data.reshape(1, -1)

            for sample in data:
                X.append(normalize(sample))
                y.append(label)
                positions.append(position)
                position += 1

    return np.array(X), np.array(y), np.array(positions)


def lag_analysis(X, y):
    """
    Measures how fast samples stop looking alike as they get further apart in the recording. For each lag it compares
    the distance between samples that are `lag` positions apart with the distance between random pairs of the same
    letter. A ratio close to 1 means the samples are already as different as random ones. This tells us how big the
    gap between train and test must be to avoid leakage
    :param
    X: np.ndarray
    Normalized samples
    y: np.ndarray
    Letter of each sample
    :return:
    None
    """
    rng = np.random.default_rng(SEED)

    print("=" * 70)
    print("REDUNDANCY BY LAG (distance at lag / distance between random pairs)")
    print("=" * 70)

    for lag in LAGS:
        ratios = []
        for label in np.unique(y):
            Xc = X[y == label]  # already in recording order
            if len(Xc) <= lag:
                continue

            lagged = np.linalg.norm(Xc[lag:] - Xc[:-lag], axis=1).mean()

            idx_a = rng.integers(0, len(Xc), 2000)
            idx_b = rng.integers(0, len(Xc), 2000)
            random_pairs = np.linalg.norm(Xc[idx_a] - Xc[idx_b], axis=1).mean()

            ratios.append(lagged / random_pairs)

        print(f"lag {lag:>4}: ratio = {np.mean(ratios):.2f}")
    print()


def random_folds(y):
    """
    Builds the classic stratified K-fold split: samples are shuffled and distributed at random. This is the LEAKY
    baseline, because near-identical neighbouring frames end up on both sides
    :param
    y: np.ndarray
    Letter of each sample
    :return:
    list
    List of (train_indices, test_indices) tuples, one per fold
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y))


def blocked_folds(y, positions):
    """
    Builds a time-based (blocked) K-fold split. For every letter, its samples are kept in recording order and cut
    into N_FOLDS contiguous blocks. In fold k, block k of every letter is the test set. Then GAP samples on each side
    of the test block are removed from training (purging), so no training sample is a direct neighbour of a test one
    :param
    y: np.ndarray
    Letter of each sample
    positions: np.ndarray
    Recording order of each sample inside its letter
    :return:
    list
    List of (train_indices, test_indices) tuples, one per fold
    """
    folds = []

    for k in range(N_FOLDS):
        train_idx, test_idx = [], []

        for label in np.unique(y):
            idx = np.where(y == label)[0]
            idx = idx[np.argsort(positions[idx])]  # make sure they are in recording order
            n = len(idx)

            start = k * n // N_FOLDS
            end = (k + 1) * n // N_FOLDS

            test_idx.extend(idx[start:end])
            train_idx.extend(idx[:max(0, start - GAP)])  # before the test block, minus the gap
            train_idx.extend(idx[min(n, end + GAP):])  # after the test block, minus the gap

        folds.append((np.array(train_idx), np.array(test_idx)))

    return folds


def evaluate(X, y, folds, name):
    """
    Trains and evaluates a Random Forest on every fold and prints accuracy and macro F1 for each one. It also
    collects the out-of-fold predictions, so every sample is predicted exactly once by a model that did not see it
    :param
    X: np.ndarray
    Normalized samples
    y: np.ndarray
    Letter of each sample
    folds: list
    List of (train_indices, test_indices) tuples
    name: str
    Name of the split strategy, used in the printed report
    :return:
    dict
    Mean and standard deviation of accuracy and macro F1, plus all true labels and predictions
    """
    print("=" * 70)
    print(f"{name.upper()} SPLIT ({N_FOLDS} folds)")
    print("=" * 70)

    accs, f1s = [], []
    y_true_all, y_pred_all = [], []

    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        model = RandomForestClassifier(n_estimators=N_ESTIMATORS, n_jobs=-1, random_state=SEED)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])

        acc = accuracy_score(y[test_idx], pred)
        f1 = f1_score(y[test_idx], pred, average="macro")
        accs.append(acc)
        f1s.append(f1)
        y_true_all.extend(y[test_idx])
        y_pred_all.extend(pred)

        print(f"fold {fold}: accuracy = {acc:.3f}   macro F1 = {f1:.3f}   "
              f"(train = {len(train_idx)}, test = {len(test_idx)})")

    print(f"MEAN  : accuracy = {np.mean(accs):.3f} ± {np.std(accs):.3f}   "
          f"macro F1 = {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print()

    return {
        "acc_mean": np.mean(accs), "acc_std": np.std(accs),
        "f1_mean": np.mean(f1s), "f1_std": np.std(f1s),
        "y_true": np.array(y_true_all), "y_pred": np.array(y_pred_all),
    }


def save_confusion_matrix(y_true, y_pred, labels, path):
    """
    Plots the confusion matrix (rows = real letter, columns = predicted letter) and saves it as an image
    :param
    y_true: np.ndarray
    Real letters
    y_pred: np.ndarray
    Predicted letters
    labels: list
    Ordered list of letters used for rows and columns
    path: str
    Where to save the image
    :return:
    np.ndarray
    The confusion matrix as a 26x26 array of counts
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(13, 11))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False
    )
    for text in ax.texts:
        text.set_fontsize(7)
    ax.set_title("Confusion matrix (blocked split, out-of-fold predictions)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return cm


def print_top_confusions(cm, labels, top=10):
    """
    Prints the most frequent mistakes: the pairs (real letter -> predicted letter) with the highest counts outside
    the diagonal of the confusion matrix
    :param
    cm: np.ndarray
    Confusion matrix
    labels: list
    Ordered list of letters
    top: int
    Number of pairs to print
    :return:
    None
    """
    errors = cm.copy()
    np.fill_diagonal(errors, 0)

    flat = np.argsort(errors, axis=None)[::-1][:top]
    print(f"Top {top} confusions (real -> predicted: count):")
    for f in flat:
        i, j = np.unravel_index(f, errors.shape)
        if errors[i, j] == 0:
            break
        print(f"  {labels[i]} -> {labels[j]}: {errors[i, j]}")
    print()


def main():
    """
    Runs the full evaluation: loads the data, analyses redundancy between neighbouring samples, compares a random
    split with a blocked split, and saves the per-letter report and the confusion matrix of the blocked split
    :return:
    None
    """
    X, y, positions = load_dataset()
    labels = sorted(np.unique(y))
    print(f"Data loaded: X = {X.shape}, {len(labels)} letters\n")

    lag_analysis(X, y)

    random_result = evaluate(X, y, random_folds(y), "random")
    blocked_result = evaluate(X, y, blocked_folds(y, positions), "blocked")

    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Split':<10}{'Accuracy':>20}{'Macro F1':>20}")
    for name, r in [("random", random_result), ("blocked", blocked_result)]:
        print(f"{name:<10}{r['acc_mean']:>12.3f} ± {r['acc_std']:.3f}{r['f1_mean']:>12.3f} ± {r['f1_std']:.3f}")
    print()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    report = classification_report(blocked_result["y_true"], blocked_result["y_pred"], labels=labels, digits=3)
    print("Per-letter report (blocked split):")
    print(report)
    with open(os.path.join(RESULTS_DIR, "classification_report_blocked.txt"), "w") as f:
        f.write(report)

    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix_blocked.png")
    cm = save_confusion_matrix(blocked_result["y_true"], blocked_result["y_pred"], labels, cm_path)
    print_top_confusions(cm, labels)

    print(f"Saved: {cm_path}")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'classification_report_blocked.txt')}")


if __name__ == "__main__":
    main()