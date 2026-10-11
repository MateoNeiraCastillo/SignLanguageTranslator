import os
import re
from collections import Counter

import numpy as np

DATASET_DIR = "dataset_landmarks"
N_FEATURES = 63  # 21 landmarks x (x, y, z)
EXPECTED_CLASSES = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
RNG = np.random.default_rng(0)  # fixed seed so the report is reproducible


def normalize(sample):
    """
    Transform the array into a 21:3 matrix and normalize it using the wrist as the origin. This is the same
    normalization used in training.py, so the distances computed here live in the same space the model sees
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
    Builds a sort key that orders file names the way a human would ("2.npy" before "10.npy"). Plain sorting is
    alphabetical and would put "10" before "2", which breaks the "consecutive files" check below
    :param
    name: str
    File name
    :return:
    list
    Mix of strings and integers used as the sort key
    """
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def load_file(path):
    """
    Loads a .npy file and always returns a 2D array, so single-frame and multi-frame files are handled the same way
    :param
    path: str
    Path to the .npy file
    :return:
    np.ndarray
    Array of shape (n_frames, n_features)
    """
    data = np.load(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def inspect_class(class_dir):
    """
    Inspects every .npy file of one letter. It counts files and frames, detects broken files and measures how
    similar consecutive samples are compared to random pairs of samples of the same letter. If consecutive samples
    are much closer than random ones, the data was recorded as a continuous stream of almost identical frames
    :param
    class_dir: str
    Path to the folder of one letter
    :return:
    dict
    Summary with the number of files, frames per file, problems found, example file names and the
    consecutive / random distance ratio
    """
    files = sorted((f for f in os.listdir(class_dir) if f.endswith(".npy")), key=natural_key)

    frames_per_file = []
    problems = []
    samples = []  # all normalized samples of this letter, in file order

    for file in files:
        path = os.path.join(class_dir, file)
        try:
            data = load_file(path)
        except Exception as e:
            problems.append(f"{file}: could not be loaded ({e})")
            continue

        if data.ndim != 2 or data.shape[1] != N_FEATURES:
            problems.append(f"{file}: unexpected shape {data.shape}")
            continue

        if np.isnan(data).any():
            problems.append(f"{file}: contains NaN values")
            continue

        frames_per_file.append(data.shape[0])
        samples.extend(normalize(sample) for sample in data)

    ratio = None
    if len(samples) > 2:
        samples = np.array(samples)

        # distance between each sample and the next one (in recording order)
        consecutive = np.linalg.norm(np.diff(samples, axis=0), axis=1).mean()

        # distance between random pairs of samples of the same letter
        idx_a = RNG.integers(0, len(samples), 2000)
        idx_b = RNG.integers(0, len(samples), 2000)
        random_pairs = np.linalg.norm(samples[idx_a] - samples[idx_b], axis=1).mean()

        if random_pairs > 0:
            ratio = consecutive / random_pairs

    return {
        "n_files": len(files),
        "frames_per_file": frames_per_file,
        "problems": problems,
        "examples": files[:3],
        "ratio": ratio,
    }


def main():
    """
    Runs the inspection over the whole dataset and prints a report: which folders exist, how many files and
    samples each letter has, whether each file is a single frame or a recording, and how redundant the data is
    :return:
    None
    """
    if not os.path.isdir(DATASET_DIR):
        print(f"Folder '{DATASET_DIR}' not found. Run this script from the project root.")
        return

    folders = sorted(d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d)))

    missing = [c for c in EXPECTED_CLASSES if c not in folders]
    unexpected = [f for f in folders if f not in EXPECTED_CLASSES]

    print("=" * 70)
    print("FOLDERS")
    print("=" * 70)
    print(f"Found {len(folders)} folders")
    print(f"Missing letters: {missing if missing else 'none'}")
    print(f"Unexpected folders (e.g. 'XX'): {unexpected if unexpected else 'none'}")

    print()
    print("=" * 70)
    print("PER LETTER")
    print("=" * 70)
    print(f"{'Letter':<8}{'Files':>7}{'Samples':>9}{'Frames/file (min/mean/max)':>30}{'Ratio':>8}  Examples")

    all_frames = []
    all_ratios = []
    all_problems = []

    for folder in folders:
        info = inspect_class(os.path.join(DATASET_DIR, folder))
        frames = info["frames_per_file"]
        all_frames.extend(frames)
        all_problems.extend(f"{folder}/{p}" for p in info["problems"])

        if frames:
            frames_txt = f"{min(frames)} / {np.mean(frames):.1f} / {max(frames)}"
        else:
            frames_txt = "-"

        ratio_txt = f"{info['ratio']:.2f}" if info["ratio"] is not None else "-"
        if info["ratio"] is not None:
            all_ratios.append(info["ratio"])

        print(f"{folder:<8}{info['n_files']:>7}{sum(frames):>9}{frames_txt:>30}{ratio_txt:>8}  {info['examples']}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files:   {len(all_frames)}")
    print(f"Total samples: {sum(all_frames)}")
    print(f"Frames per file distribution (frames: number of files): {dict(sorted(Counter(all_frames).items()))}")

    if all_problems:
        print(f"\n{len(all_problems)} problematic files:")
        for p in all_problems:
            print(f"  - {p}")
    else:
        print("No broken files found")

    print()
    if all_frames and max(all_frames) == 1:
        print("-> Every file is a SINGLE frame. Splitting by file does NOT prevent leakage:")
        print("   neighbouring files are probably frames of the same recording. Check the file")
        print("   names above to see if they reveal sessions (timestamps, prefixes...).")
    elif all_frames:
        print("-> Files contain SEVERAL frames. Each file is probably one recording, so a")
        print("   split BY FILE (all frames of a file on the same side) is possible.")

    if all_ratios:
        mean_ratio = np.mean(all_ratios)
        print(f"\nMean consecutive/random distance ratio: {mean_ratio:.2f}")
        print("   Close to 1.0 -> consecutive samples are as different as random ones (independent poses).")
        print("   Well below 1.0 (e.g. < 0.3) -> consecutive samples are near-duplicates (continuous recording),")
        print("   so a random train/test split would put almost identical frames on both sides.")


if __name__ == "__main__":
    main()