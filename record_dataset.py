import argparse
import os
import random
import re
import string
import time

import cv2
import mediapipe as mp
import numpy as np

OUTPUT_DIR = "dataset_v2"  # new dataset, kept separate from the original dataset_landmarks/
COUNTDOWN = 3.0  # seconds to get into position before a take starts recording
MAX_HAND_LOSS = 0.5  # if the hand disappears for longer than this during a take, the take is discarded
WINDOW_NAME = "Dataset Recorder"
FONT = cv2.FONT_HERSHEY_SIMPLEX

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def parse_args():
    """
    Reads the options given in the terminal, so the same script can be used for any signer and session without
    editing the code (e.g. python record_dataset.py --signer s02 --session 1)
    :return:
    argparse.Namespace
    Object with the attributes signer, session, letters, takes, frames and shuffle
    """
    parser = argparse.ArgumentParser(description="Record hand landmark takes for the sign language dataset")
    parser.add_argument("--signer", required=True,
                        help="Short id of the person signing, e.g. s01. Never use real names")
    parser.add_argument("--session", type=int, default=1,
                        help="Session number. Use a new one for a new day, place or lighting")
    parser.add_argument("--letters", default=string.ascii_uppercase,
                        help="Letters to record, e.g. UVR (default: A-Z)")
    parser.add_argument("--takes", type=int, default=5,
                        help="Takes to record per letter in this run (default: 5)")
    parser.add_argument("--frames", type=int, default=30,
                        help="Frames with a detected hand saved per take (default: 30, about 1 second)")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True,
                        help="Record the letters in random order (default). Use --no-shuffle for A-Z order")
    return parser.parse_args()


def existing_takes(letter_dir, prefix):
    """
    Finds the take numbers already saved for one signer, session and letter. This lets you stop the script and
    resume later without overwriting earlier takes
    :param
    letter_dir: str
    Folder where the takes of this letter are saved
    prefix: str
    Start of the file names of this signer, session and letter (e.g. "s01_ses1_A_t")
    :return:
    list
    Take numbers already saved
    """
    pattern = re.compile(re.escape(prefix) + r"(\d+)_[A-Z]\.npy$")
    numbers = []
    for file in os.listdir(letter_dir):
        match = pattern.match(file)
        if match:
            numbers.append(int(match.group(1)))
    return numbers


def read_frame(cap, hands):
    """
    Reads one frame from the camera and runs MediaPipe on it. The frame is NOT mirrored, so the landmarks are
    in the same orientation as in main.py
    :param
    cap: cv2.VideoCapture
    Open camera
    hands: mediapipe Hands object
    Hand detector
    :return:
    tuple
    (frame, results), or (None, None) if the camera returned nothing
    """
    ret, frame = cap.read()
    if not ret:
        return None, None
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frame, results


def show(frame, results, lines):
    """
    Displays the frame with the hand skeleton and some lines of text. Only the PREVIEW is mirrored, because a
    mirrored image feels natural (like a mirror); the saved landmarks always come from the original frame
    :param
    frame: np.ndarray
    Original camera frame
    results: mediapipe results
    Output of hands.process for this frame
    lines: list
    Lines of text to draw on the top left corner
    :return:
    int
    Code of the key pressed (255 if none)
    """
    display = frame.copy()
    if results.multi_hand_landmarks:
        mp_draw.draw_landmarks(display, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    display = cv2.flip(display, 1)  # mirror the preview only

    for i, line in enumerate(lines):
        y = 35 + 32 * i
        cv2.putText(display, line, (20, y), FONT, 0.7, (0, 0, 0), 4)  # black outline for readability
        cv2.putText(display, line, (20, y), FONT, 0.7, (255, 255, 255), 2)

    cv2.imshow(WINDOW_NAME, display)
    return cv2.waitKey(1) & 0xFF


def record_take(cap, hands, n_frames, header):
    """
    Records one take: a countdown to get into position, then n_frames consecutive frames with a detected hand.
    The take is discarded if the hand disappears for too long, so every saved take is a clean, continuous recording
    :param
    cap: cv2.VideoCapture
    Open camera
    hands: mediapipe Hands object
    Hand detector
    n_frames: int
    Number of frames with a detected hand to save
    header: list
    Lines of text shown on top of the countdown and the recording messages
    :return:
    tuple
    (landmarks, hand) with an array of shape (n_frames, 63) and "L" or "R", or (None, reason) if it was discarded
    """
    start = time.time()
    while True:
        frame, results = read_frame(cap, hands)
        if frame is None:
            return None, "camera error"
        remaining = COUNTDOWN - (time.time() - start)
        if remaining <= 0:
            break
        key = show(frame, results, header + [f"Get ready... {remaining:.1f}"])
        if key == ord("q"):
            return None, "cancelled"

    landmarks, hand_labels = [], []
    last_seen = time.time()

    while len(landmarks) < n_frames:
        frame, results = read_frame(cap, hands)
        if frame is None:
            return None, "camera error"
        now = time.time()

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            landmarks.append([value for lm in hand.landmark for value in (lm.x, lm.y, lm.z)])
            if results.multi_handedness:
                hand_labels.append(results.multi_handedness[0].classification[0].label[0])  # "L" or "R"
            last_seen = now
        elif now - last_seen > MAX_HAND_LOSS:
            return None, "hand lost"

        key = show(frame, results, header + [f"RECORDING {len(landmarks)}/{n_frames}"])
        if key == ord("q"):
            return None, "cancelled"

    # MediaPipe's label is computed on the non-mirrored frame, so it may be swapped (left <-> right).
    # That is fine: we only need it to be CONSISTENT, to tell apart people who sign with different hands
    hand = max(set(hand_labels), key=hand_labels.count) if hand_labels else "U"
    return np.array(landmarks), hand


def main():
    """
    Runs the recorder. For each letter it records the requested number of takes, saving each one as a separate
    .npy file of shape (frames, 63) with the RAW landmarks. By default the letters are recorded in random order:
    with a fixed A-Z order, the last letters would always be recorded when the person is most tired, and the model
    could learn that "tired hand" means "Z" (a confounder). The file name stores the metadata:
    dataset_v2/<signer>/<letter>/<signer>_ses<session>_<letter>_t<take>_<hand>.npy
    Keys: SPACE = record a take, D = delete the last take of this letter, N = next letter, Q = quit
    :return:
    None
    """
    args = parse_args()
    letters = [c for c in args.letters.upper() if c in string.ascii_uppercase]
    if args.shuffle:
        random.shuffle(letters)
    print("Recording order:", " ".join(letters))

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,  # same settings as main.py, so the data looks like what the app sees
        min_tracking_confidence=0.6
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error")
        return

    saved = []  # files saved in this run, used by the delete key
    message = ""
    quit_all = False

    for letter_number, letter in enumerate(letters, start=1):
        if quit_all:
            break

        letter_dir = os.path.join(OUTPUT_DIR, args.signer, letter)
        os.makedirs(letter_dir, exist_ok=True)
        prefix = f"{args.signer}_ses{args.session}_{letter}_t"
        done = 0

        while done < args.takes:
            frame, results = read_frame(cap, hands)
            if frame is None:
                quit_all = True
                break

            n_saved = len(existing_takes(letter_dir, prefix))
            header = [
                f"Signer {args.signer} | session {args.session}",
                f"Letter {letter} ({letter_number}/{len(letters)}) | take {done + 1}/{args.takes} | "
                f"already saved: {n_saved}",
            ]
            lines = header + ["SPACE record | D delete last | N next letter | Q quit"]
            if message:
                lines.append(message)

            key = show(frame, results, lines)

            if key == ord(" "):
                data, info = record_take(cap, hands, args.frames, header)
                if data is None:
                    message = f"Take discarded ({info})"
                    if info == "camera error":
                        quit_all = True
                        break
                else:
                    take = max(existing_takes(letter_dir, prefix), default=0) + 1
                    path = os.path.join(letter_dir, f"{prefix}{take:02d}_{info}.npy")
                    np.save(path, data)
                    saved.append(path)
                    done += 1
                    message = f"Saved {os.path.basename(path)} - move a bit (distance, angle) before the next take"

            elif key == ord("d"):
                if saved and os.path.dirname(saved[-1]) == letter_dir:
                    removed = saved.pop()
                    os.remove(removed)
                    done = max(0, done - 1)
                    message = f"Deleted {os.path.basename(removed)}"
                else:
                    message = "Nothing to delete for this letter"

            elif key == ord("n"):
                message = f"Skipped the rest of letter {letter}"
                break

            elif key == ord("q"):
                quit_all = True
                break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    print(f"\nTakes saved in this run: {len(saved)}")
    for path in saved:
        print("  ", path)


if __name__ == "__main__":
    main()