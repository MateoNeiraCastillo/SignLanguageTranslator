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

from features import extract_features, normalize

DATASET_DIR = "dataset_landmarks"
RESULTS_DIR = "results"

N_FOLDS = 5  # number of folds for cross-validation
GAP = 20  # samples removed from training on each side of a test block (purging)
N_ESTIMATORS = 500  # same as training.py, so we evaluate the same kind of model
SEED = 42  # fixed seed so results are reproducible
LAGS = [1, 2, 5, 10, 20, 50, 100]  # distances (in samples) used in the redundancy analysis


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
    Loads every raw sample of the dataset and keeps track of its position inside its letter. The position is the
    recording order, which is what we use to build time-based (blocked) splits. Samples are returned raw so that
    different feature extractors can be compared on exactly the same data
    :return:
    tuple(np.ndarray, np.ndarray, np.ndarray)
    X with the raw samples, y with the letters and positions with the recording order inside each letter
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
                X.append(sample)
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
    Feature matrix
    y: np.ndarray
    Letter of each sample
    folds: list
    List of (train_indices, test_indices) tuples
    name: str
    Name of the experiment, used in the printed report
    :return:
    dict
    Per-fold accuracies, mean and standard deviation of accuracy and macro F1, plus all true labels and predictions
    """
    print("=" * 70)
    print(f"{name.upper()} ({N_FOLDS} folds, {X.shape[1]} features)")
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
        "accs": np.array(accs),
        "acc_mean": np.mean(accs), "acc_std": np.std(accs),
        "f1_mean": np.mean(f1s), "f1_std": np.std(f1s),
        "y_true": np.array(y_true_all), "y_pred": np.array(y_pred_all),
    }


def save_confusion_matrix(y_true, y_pred, labels, path, title):
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
    title: str
    Title of the plot
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
    ax.set_title(title)
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


def save_results(result, labels, suffix, title):
    """
    Saves the per-letter report and the confusion matrix of one experiment to the results folder and prints the
    most frequent confusions
    :param
    result: dict
    Output of evaluate()
    labels: list
    Ordered list of letters
    suffix: str
    Text added to the file names to tell experiments apart ("" for the baseline)
    title: str
    Title of the confusion matrix plot
    :return:
    None
    """
    report = classification_report(result["y_true"], result["y_pred"], labels=labels, digits=3)
    report_path = os.path.join(RESULTS_DIR, f"classification_report_blocked{suffix}.txt")
    with open(report_path, "w") as f:
        f.write(report)

    cm_path = os.path.join(RESULTS_DIR, f"confusion_matrix_blocked{suffix}.png")
    cm = save_confusion_matrix(result["y_true"], result["y_pred"], labels, cm_path, title)

    print(f"[{title}]")
    print_top_confusions(cm, labels)
    print(f"Saved: {cm_path}")
    print(f"Saved: {report_path}\n")


def compare_per_letter(baseline, improved, labels):
    """
    Prints the F1 of every letter with the baseline features and with the geometric features, and the difference.
    This shows whether the new features fix the letters they were designed for (U/V/R, T/F) without hurting others
    :param
    baseline: dict
    Output of evaluate() with the baseline features
    improved: dict
    Output of evaluate() with the geometric features
    labels: list
    Ordered list of letters
    :return:
    None
    """
    f1_base = f1_score(baseline["y_true"], baseline["y_pred"], labels=labels, average=None)
    f1_new = f1_score(improved["y_true"], improved["y_pred"], labels=labels, average=None)

    print("=" * 70)
    print("PER-LETTER F1 (blocked split): baseline vs geometric features")
    print("=" * 70)
    print(f"{'Letter':<8}{'Baseline':>10}{'Geometric':>11}{'Change':>9}")
    for letter, a, b in zip(labels, f1_base, f1_new):
        mark = "  <-- " if abs(b - a) >= 0.03 else ""
        print(f"{letter:<8}{a:>10.3f}{b:>11.3f}{b - a:>+9.3f}{mark}")
    print()


def main():
    """
    Runs the full evaluation: loads the data, analyses redundancy between neighbouring samples, compares a random
    split with a blocked split, and compares the baseline features with the geometric features using exactly the same
    blocked folds. Saves the per-letter reports and confusion matrices of both blocked experiments
    :return:
    None
    """
    X_raw, y, positions = load_dataset()
    labels = sorted(np.unique(y))

    X_base = np.array([normalize(sample) for sample in X_raw])
    X_geo = np.array([extract_features(sample) for sample in X_raw])
    print(f"Data loaded: {len(y)} samples, {len(labels)} letters")
    print(f"Baseline features: {X_base.shape[1]}   Geometric features: {X_geo.shape[1]}\n")

    lag_analysis(X_base, y)

    # The same folds are used for both feature sets, so the comparison is paired: only the features change
    folds_random = random_folds(y)
    folds_blocked = blocked_folds(y, positions)

    random_base = evaluate(X_base, y, folds_random, "random split - baseline")
    blocked_base = evaluate(X_base, y, folds_blocked, "blocked split - baseline")
    blocked_geo = evaluate(X_geo, y, folds_blocked, "blocked split - geometric features")

    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Experiment':<32}{'Accuracy':>18}{'Macro F1':>18}")
    for name, r in [("random - baseline", random_base),
                    ("blocked - baseline", blocked_base),
                    ("blocked - geometric", blocked_geo)]:
        print(f"{name:<32}{r['acc_mean']:>10.3f} ± {r['acc_std']:.3f}{r['f1_mean']:>10.3f} ± {r['f1_std']:.3f}")

    diffs = blocked_geo["accs"] - blocked_base["accs"]
    print("\nAccuracy change per fold (geometric - baseline): "
          + "  ".join(f"{d:+.3f}" for d in diffs))
    print(f"Folds improved: {np.sum(diffs > 0)} / {N_FOLDS}\n")

    compare_per_letter(blocked_base, blocked_geo, labels)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_results(blocked_base, labels, "", "Confusion matrix - baseline (blocked split)")
    save_results(blocked_geo, labels, "_features", "Confusion matrix - geometric features (blocked split)")


if __name__ == "__main__":
    main()