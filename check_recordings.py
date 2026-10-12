import argparse
import os
import re
import string
from collections import Counter, defaultdict

import joblib
import numpy as np

from features import extract_features

DATASET_DIR = "dataset_v2"
MODEL_PATH = "model/modelo_senas.pkl"
N_FEATURES = 63  # 21 landmarks x (x, y, z)
AGREEMENT_THRESHOLD = 0.5  # a take is flagged if less than half of its frames are predicted as its own letter
MOVEMENT_LETTERS = {"J", "Z"}  # recorded with movement, so the static model is expected to struggle with them

# signer_ses<session>_<letter>_t<take>_<hand>.npy  (e.g. s02_ses1_A_t03_R.npy)
FILE_PATTERN = re.compile(
    r"^(?P<signer>[A-Za-z0-9]+)_ses(?P<session>\d+)_(?P<letter>[A-Z])_t(?P<take>\d+)_(?P<hand>[A-Z])\.npy$"
)


def parse_args():
    """
    Reads the options given in the terminal, so the check can be limited to one signer or session
    :return:
    argparse.Namespace
    Object with the attributes signer, session, takes, frames and no_model
    """
    parser = argparse.ArgumentParser(description="Quality check of the takes recorded with record_dataset.py")
    parser.add_argument("--signer", help="Only check this signer (default: all)")
    parser.add_argument("--session", type=int, help="Only check this session (default: all)")
    parser.add_argument("--takes", type=int, default=5, help="Expected takes per letter and session (default: 5)")
    parser.add_argument("--frames", type=int, default=30, help="Expected frames per take (default: 30)")
    parser.add_argument("--no-model", action="store_true", help="Skip the check with the trained model")
    return parser.parse_args()


def scan_files(signer_filter, session_filter):
    """
    Walks dataset_v2/<signer>/<letter>/ and reads the metadata of every take from its file name. Files whose name
    does not follow the expected format, or that are stored in the folder of another letter, are reported as problems
    :param
    signer_filter: str or None
    Only keep takes of this signer
    session_filter: int or None
    Only keep takes of this session
    :return:
    tuple(list, list)
    List of takes (dicts with path, signer, session, letter, take and hand) and list of problems found
    """
    takes, problems = [], []

    for signer in sorted(os.listdir(DATASET_DIR)):
        signer_dir = os.path.join(DATASET_DIR, signer)
        if not os.path.isdir(signer_dir) or (signer_filter and signer != signer_filter):
            continue

        for folder in sorted(os.listdir(signer_dir)):
            letter_dir = os.path.join(signer_dir, folder)
            if not os.path.isdir(letter_dir):
                continue

            for file in sorted(os.listdir(letter_dir)):
                path = os.path.join(letter_dir, file)
                match = FILE_PATTERN.match(file)

                if not match:
                    problems.append(f"{path}: unexpected file name")
                    continue

                info = match.groupdict()
                info["session"] = int(info["session"])
                info["take"] = int(info["take"])
                info["path"] = path

                if session_filter is not None and info["session"] != session_filter:
                    continue
                if info["signer"] != signer or info["letter"] != folder:
                    problems.append(f"{path}: name does not match its folder (signer or letter)")
                    continue

                takes.append(info)

    return takes, problems


def check_files(takes, expected_frames):
    """
    Loads every take and checks that it has the expected shape and no missing values. Broken takes are removed
    from the list so the next checks only use valid data
    :param
    takes: list
    Takes returned by scan_files
    expected_frames: int
    Number of frames each take should have
    :return:
    tuple(list, list)
    Valid takes (with their landmarks added under the key "data") and list of problems found
    """
    valid, problems = [], []

    for take in takes:
        try:
            data = np.load(take["path"])
        except Exception as e:
            problems.append(f"{take['path']}: could not be loaded ({e})")
            continue

        if data.ndim != 2 or data.shape[1] != N_FEATURES:
            problems.append(f"{take['path']}: unexpected shape {data.shape}")
            continue
        if data.shape[0] != expected_frames:
            problems.append(f"{take['path']}: {data.shape[0]} frames instead of {expected_frames}")
        if np.isnan(data).any():
            problems.append(f"{take['path']}: contains NaN values")
            continue

        take["data"] = data
        valid.append(take)

    return valid, problems


def check_coverage(takes, expected_takes):
    """
    Counts the takes of every signer, session and letter and reports letters that are missing or have fewer takes
    than expected. It also reports which hand each signer used, so a mix of left and right hands is easy to spot
    :param
    takes: list
    Valid takes
    expected_takes: int
    Takes expected per letter and session
    :return:
    list
    Lines of text describing the coverage problems found
    """
    lines = []
    counts = defaultdict(Counter)  # (signer, session) -> Counter(letter -> takes)
    hands = defaultdict(Counter)  # signer -> Counter(hand -> takes)

    for take in takes:
        counts[(take["signer"], take["session"])][take["letter"]] += 1
        hands[take["signer"]][take["hand"]] += 1

    for (signer, session), letter_counts in sorted(counts.items()):
        missing = [c for c in string.ascii_uppercase if letter_counts[c] == 0]
        short = [f"{c}({letter_counts[c]})" for c in string.ascii_uppercase if 0 < letter_counts[c] < expected_takes]
        total = sum(letter_counts.values())

        lines.append(f"{signer} session {session}: {total} takes, {len(letter_counts)} letters")
        if missing:
            lines.append(f"   missing letters: {' '.join(missing)}")
        if short:
            lines.append(f"   fewer than {expected_takes} takes: {' '.join(short)}")

    for signer, hand_counts in sorted(hands.items()):
        summary = ", ".join(f"{hand}: {n}" for hand, n in sorted(hand_counts.items()))
        warning = "   <-- mixed hands, check this signer" if len(hand_counts) > 1 else ""
        lines.append(f"{signer} hands -> {summary}{warning}")

    return lines


def check_with_model(takes, model):
    """
    Predicts every frame of every take with the current model and flags the takes where most frames are predicted
    as a different letter. A flagged take is not necessarily wrong (the model was trained on one person), but it is
    worth reviewing: it may be the wrong letter, a badly formed sign or a recording problem. J and Z are reported
    separately because they are recorded with movement and the static model is expected to struggle with them
    :param
    takes: list
    Valid takes
    model: sklearn classifier
    Trained model
    :return:
    tuple(list, float)
    List of flagged takes (dicts with path, letter, predicted letter and agreement) and the overall frame accuracy
    """
    flagged = []
    correct, total = 0, 0

    for take in takes:
        features = np.array([extract_features(frame) for frame in take["data"]])
        predictions = model.predict(features)

        agreement = np.mean(predictions == take["letter"])
        correct += np.sum(predictions == take["letter"])
        total += len(predictions)

        if agreement < AGREEMENT_THRESHOLD:
            most_common = Counter(predictions).most_common(1)[0][0]
            flagged.append({
                "path": take["path"],
                "letter": take["letter"],
                "predicted": most_common,
                "agreement": agreement,
            })

    accuracy = correct / total if total else 0
    return flagged, accuracy


def load_model():
    """
    Loads the trained model and checks that it uses the same features that features.py produces
    :return:
    sklearn classifier or None
    The model, or None if it is missing or was trained with different features
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Run python training.py first, or use --no-model.")
        return None

    model = joblib.load(MODEL_PATH)
    expected = len(extract_features(np.zeros(N_FEATURES)))
    if model.n_features_in_ != expected:
        print(f"The model expects {model.n_features_in_} features but features.py produces {expected}. "
              f"Retrain it with: python training.py")
        return None
    return model


def main():
    """
    Runs every quality check on the recorded takes and prints a report: broken or misnamed files, missing letters
    or takes, hands used by each signer, and the takes that the current model predicts as a different letter
    :return:
    None
    """
    args = parse_args()

    if not os.path.isdir(DATASET_DIR):
        print(f"Folder '{DATASET_DIR}' not found. Record something first with record_dataset.py.")
        return

    takes, problems = scan_files(args.signer, args.session)
    takes, file_problems = check_files(takes, args.frames)
    problems += file_problems

    print("=" * 70)
    print("FILES")
    print("=" * 70)
    print(f"Valid takes: {len(takes)}")
    if problems:
        print(f"{len(problems)} problems:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("No broken or misnamed files")

    print()
    print("=" * 70)
    print("COVERAGE")
    print("=" * 70)
    for line in check_coverage(takes, args.takes):
        print(line)

    if args.no_model or not takes:
        return

    model = load_model()
    if model is None:
        return

    flagged, accuracy = check_with_model(takes, model)
    movement = [f for f in flagged if f["letter"] in MOVEMENT_LETTERS]
    static = [f for f in flagged if f["letter"] not in MOVEMENT_LETTERS]

    print()
    print("=" * 70)
    print("CHECK WITH THE CURRENT MODEL")
    print("=" * 70)
    print(f"Frames predicted as their own letter: {accuracy:.1%}")
    print("(The model was trained on one person, so a lower number for new signers is expected.)")

    print(f"\nTakes to review (less than {AGREEMENT_THRESHOLD:.0%} of frames predicted as their letter): {len(static)}")
    for f in sorted(static, key=lambda f: f["agreement"]):
        print(f"  {f['path']}: labelled {f['letter']}, predicted {f['predicted']} "
              f"({f['agreement']:.0%} agreement)")

    if movement:
        print(f"\nJ/Z takes flagged (expected, they include movement): {len(movement)}")

    if static:
        print("\nFor each take to review: was it the right letter and well formed? If not, delete the file and")
        print("record that letter again, e.g.: python record_dataset.py --signer s02 --session 1 --letters U --takes 1")


if __name__ == "__main__":
    main()
