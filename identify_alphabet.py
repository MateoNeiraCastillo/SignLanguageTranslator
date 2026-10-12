import os

import matplotlib
matplotlib.use("Agg")  # save figures to disk without opening a window
import matplotlib.pyplot as plt
import numpy as np

from features import normalize

DATASET_DIR = "dataset_landmarks"
RESULTS_DIR = "results"
FIGURE_PATH = os.path.join(RESULTS_DIR, "letter_poses.png")

# Pairs of landmarks joined by a bone, used to draw the hand skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # base of the palm
]

# (middle joint, tip) of each finger except the thumb: I = index, M = middle, R = ring, P = pinky
FINGERS = {"I": (6, 8), "M": (10, 12), "R": (14, 16), "P": (18, 20)}

# Which fingers (index, middle, ring, pinky) are extended in each letter of the ASL alphabet.
# "." means folded. None means the handshape is curved or hooked (C, O, X), so this simple test cannot judge it
ASL_EXTENDED = {
    "A": "....", "B": "IMRP", "C": None, "D": "I...", "E": "....", "F": ".MRP", "G": "I...",
    "H": "IM..", "I": "...P", "J": "...P", "K": "IM..", "L": "I...", "M": "....", "N": "....",
    "O": None, "P": "IM..", "Q": "I...", "R": "IM..", "S": "....", "T": "....", "U": "IM..",
    "V": "IM..", "W": "IMR.", "X": None, "Y": "...P", "Z": "I...",
}


def load_letters():
    """
    Loads every sample of the dataset, normalizes it and groups the samples by letter
    :return:
    dict
    Maps each letter to an array of shape (n_samples, 21, 3) with its normalized landmarks
    """
    letters = {}

    for label in sorted(os.listdir(DATASET_DIR)):
        class_dir = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(class_dir):
            continue

        samples = []
        for file in sorted(os.listdir(class_dir)):
            if file.endswith(".npy"):
                data = np.load(os.path.join(class_dir, file))
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                samples.extend(normalize(sample).reshape(21, 3) for sample in data)

        letters[label] = np.array(samples)

    return letters


def extended_pattern(pts):
    """
    Decides which fingers are extended in one hand. A finger counts as extended when its tip is further from the
    wrist than its middle joint: when a finger folds, the tip comes back towards the palm and ends up closer.
    The test uses distances, not directions, so it works whether the finger points up, sideways or down
    :param
    pts: np.ndarray
    Array of shape (21, 3) with the normalized landmarks (wrist at the origin)
    :return:
    str
    Four characters for index, middle, ring and pinky: the finger letter if extended, "." if folded (e.g. "IM..")
    """
    pattern = ""
    for name, (pip, tip) in FINGERS.items():
        extended = np.linalg.norm(pts[tip]) > np.linalg.norm(pts[pip])
        pattern += name if extended else "."
    return pattern


def letter_pattern(samples):
    """
    Computes the typical extended-finger pattern of a letter by majority vote, finger by finger, over all its samples
    :param
    samples: np.ndarray
    Array of shape (n_samples, 21, 3) with the normalized landmarks of one letter
    :return:
    tuple(str, float)
    The majority pattern and the fraction of samples whose pattern is exactly that one (how consistent the letter is)
    """
    patterns = [extended_pattern(pts) for pts in samples]

    majority = ""
    for position, name in enumerate(FINGERS):
        votes = sum(1 for p in patterns if p[position] != ".")
        majority += name if votes > len(patterns) / 2 else "."

    consistency = sum(1 for p in patterns if p == majority) / len(patterns)
    return majority, consistency


def representative(samples):
    """
    Picks the most typical sample of a letter: the one closest to the average pose. Drawing a real sample is better
    than drawing the average itself, because averaging different poses can create a hand that never existed
    :param
    samples: np.ndarray
    Array of shape (n_samples, 21, 3) with the normalized landmarks of one letter
    :return:
    np.ndarray
    Array of shape (21, 3) with the landmarks of the chosen sample
    """
    mean = samples.mean(axis=0)
    distances = np.linalg.norm((samples - mean).reshape(len(samples), -1), axis=1)
    return samples[np.argmin(distances)]


def plot_letters(letters, path):
    """
    Draws the representative hand skeleton of every letter in a grid and saves it as an image. The y axis is
    inverted because in images y grows downwards
    :param
    letters: dict
    Maps each letter to its array of normalized landmarks
    path: str
    Where to save the image
    :return:
    None
    """
    labels = sorted(letters)
    cols = 6
    rows = int(np.ceil(len(labels) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))

    for ax, label in zip(axes.flat, labels):
        pts = representative(letters[label])

        for a, b in HAND_CONNECTIONS:
            ax.plot([pts[a, 0], pts[b, 0]], [-pts[a, 1], -pts[b, 1]], color="#4a5568", linewidth=1.5)

        ax.scatter(pts[:, 0], -pts[:, 1], s=8, color="#2b6cb0", zorder=3)
        tips = [4, 8, 12, 16, 20]
        ax.scatter(pts[tips, 0], -pts[tips, 1], s=18, color="#c05621", zorder=4)  # fingertips in orange

        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.set_aspect("equal")
        ax.axis("off")

    for ax in list(axes.flat)[len(labels):]:
        ax.axis("off")

    fig.suptitle("Representative hand pose per letter (fingertips in orange)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    """
    Compares the extended-finger pattern of every letter in the dataset with the ASL alphabet, prints how many
    letters match, and saves a figure with the hand pose of every letter so it can be checked visually
    :return:
    None
    """
    letters = load_letters()

    print("=" * 70)
    print("EXTENDED FINGERS PER LETTER (I = index, M = middle, R = ring, P = pinky)")
    print("=" * 70)
    print(f"{'Letter':<8}{'Dataset':>9}{'Consistency':>13}{'ASL':>8}   Match")

    checked, matched = 0, 0
    mismatches = []

    for label in sorted(letters):
        pattern, consistency = letter_pattern(letters[label])
        expected = ASL_EXTENDED.get(label)

        if expected is None:
            result = "skipped (curved/hooked shape)"
            expected_txt = "-"
        else:
            checked += 1
            expected_txt = expected
            if pattern == expected:
                matched += 1
                result = "yes"
            else:
                result = "NO"
                mismatches.append(label)

        print(f"{label:<8}{pattern:>9}{consistency:>12.0%}{expected_txt:>8}   {result}")

    print()
    print(f"Letters that match ASL: {matched} / {checked}")
    if mismatches:
        print(f"Letters that do NOT match: {', '.join(mismatches)}")

    rate = matched / checked if checked else 0
    print()
    if rate >= 0.85:
        print("-> Very likely ASL.")
    elif rate >= 0.6:
        print("-> Partly matches ASL. Check the figure: it may be ASL with some letters recorded differently,")
        print("   or an alphabet that shares many handshapes with ASL.")
    else:
        print("-> Probably NOT ASL. Compare the figure with the LSE or LSEC alphabet charts.")

    print(f"\nFolders found: {', '.join(sorted(letters))}")
    print("(The Spanish-language alphabets usually include extra letters such as Ñ.)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_letters(letters, FIGURE_PATH)
    print(f"\nSaved: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
