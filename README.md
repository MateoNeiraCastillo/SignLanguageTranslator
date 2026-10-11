# Sign Language Translator

Real-time sign language alphabet translator using a webcam. It detects the hand with **MediaPipe**, classifies the sign with a **Random Forest**, builds words letter by letter, and reads each finished word out loud.

<!-- TODO: add a GIF or screenshot of the app running -->
<!-- ![demo](demo.gif) -->

## Features

- Real-time recognition of the 26 letters of the <!-- TODO: ASL / LSEC / ... --> alphabet
- Letters are confirmed only after the prediction stays stable for 0.5 s, to filter out noise
- Removing your hand from the frame for 0.3 s inserts a space and ends the word
- Each finished word is spoken out loud with offline text-to-speech, in a separate thread so the camera never freezes

## How it works

```
Webcam frame → MediaPipe Hands (21 landmarks × x,y,z) → normalization → Random Forest → letter
```

1. **Landmark extraction**: MediaPipe detects 21 hand landmarks, each with (x, y, z) coordinates, giving 63 features.
2. **Normalization**: coordinates are moved so the wrist is the origin, then scaled by the largest absolute value. This makes the prediction independent of where the hand is in the frame and how far it is from the camera.
3. **Classification**: a Random Forest with 500 trees predicts the letter.
4. **Text logic**: stable predictions become letters, hand absence becomes spaces, and finished words are sent to the speech queue.

## Results

**Dataset**: 26 classes (A–Z), 8,060 samples (~310 per letter), recorded by one person.

### Why the first number was misleading

Each letter was recorded as a continuous stream of webcam frames, so consecutive samples are almost identical. Measuring the distance between samples `k` positions apart, relative to random pairs of the same letter:

| Lag (samples) | 1 | 2 | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| Distance ratio | 0.38 | 0.56 | 0.80 | 0.89 | 0.93 | 0.99 | 1.05 |

A ratio well below 1 means near-duplicates. With a random train/test split (or the Random Forest's out-of-bag estimate), almost every test sample has a near-identical neighbour in the training set. This is **data leakage**, and it inflates the score.

### Leakage-free evaluation

To fix it, `evaluate.py` uses **blocked 5-fold cross-validation with purging**:

- Each letter's samples are kept in recording order and cut into 5 contiguous blocks. In each fold, one block per letter is the test set.
- The 20 samples on each side of the test block are removed from training (purging), so no training sample is a direct neighbour of a test sample. The gap of 20 was chosen from the lag analysis above (ratio ≈ 0.93).

| Evaluation | Accuracy | Macro F1 |
|---|---|---|
| Out-of-bag (Random Forest) | 99.2% | — |
| Random 5-fold CV (leaky) | 99.0% ± 0.2 | 0.990 ± 0.002 |
| **Blocked 5-fold CV + purging (gap = 20)** | **94.4% ± 1.9** | **0.943 ± 0.019** |

The result is stable with respect to the gap size (95.0% with gap = 10, 94.4% with gap = 20).

**This is still a signer-dependent, same-session estimate**: the same person, lighting and camera appear in training and test. It is an upper bound on how the model would perform with new users.

### Error analysis

22 of the 26 letters reach an F1 above 0.95. The errors are concentrated in two groups:

| Group | F1 | Most frequent confusions (blocked CV) |
|---|---|---|
| U / V / R | U 0.73 · V 0.69 · R 0.92 | V→U 80, U→V 77, R→V 14, V→R 13, R→U 12 |
| T / F | T 0.77 · F 0.77 | T→F 85, F→T 57 |

U, V and R differ mostly in the relative position of the index and middle fingers, a small difference that is hard to pick up from raw coordinates alone.

![Confusion matrix](results/confusion_matrix_blocked.png)

The full per-letter report is in [`results/classification_report_blocked.txt`](results/classification_report_blocked.txt).

## Project structure

```
├── main.py               # The translator app (camera, prediction, text and audio)
├── training.py           # Trains the final Random Forest on the full dataset
├── evaluate.py           # Leakage-free evaluation: blocked CV, per-letter metrics, confusion matrix
├── inspect_dataset.py    # Checks the dataset structure and how redundant consecutive samples are
├── coordinates.py        # Debug tool: shows the hand landmarks and their coordinates live
├── model/
│   └── modelo_senas.pkl  # Trained model
├── dataset_landmarks/    # Dataset: one folder per letter with one .npy file per frame
├── results/              # Evaluation outputs (confusion matrix, per-letter report)
└── requirements.txt
```

## Installation

Requires **Python 3.11** and a webcam.

```bash
git clone https://github.com/MateoNeiraCastillo/SignLanguageTranslator.git
cd SignLanguageTranslator

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

The camera window opens directly. Start signing.

| Key | Action |
|-----|--------|
| `t` | Resume translating |
| `s` | Pause translating |
| `q` | Quit |

Tips:
- Hold each letter steady for about half a second until it appears.
- To finish a word, take your hand out of the frame. The word will be read out loud.
- Everything runs offline. Text-to-speech uses the system voice: a persistent `NSSpeechSynthesizer` on macOS (via `pyobjc`, so each word plays almost instantly) and `pyttsx3` on Windows and Linux (on Linux you may need `sudo apt install espeak`).
- On macOS, allow camera access for your terminal or IDE the first time.

## Evaluating and retraining

The dataset is included in `dataset_landmarks/`.

To evaluate the model without leakage (takes a few minutes, it trains 10 forests):

```bash
python evaluate.py
```

This prints the lag analysis, compares random and blocked cross-validation, and saves the per-letter report and the confusion matrix to `results/`.

To train the final model used by the app:

```bash
python training.py
```

This trains the Random Forest on the full dataset and saves it to `model/modelo_senas.pkl`. The out-of-bag accuracy it prints is optimistic because of the leakage explained above; use `evaluate.py` for the real estimate.

## Limitations

- Only static letters: signs with movement (such as J and Z) are learned as a single hand pose. Their high scores reflect that pose, not the real movement.
- One hand at a time.
- The dataset was recorded by one person in one setting, so the 94.4% does not measure performance on other people, lighting or camera angles.

## Roadmap

- [x] Leakage-free evaluation: blocked cross-validation with purging, per-letter metrics and confusion matrix
- [ ] Features that capture finger relationships (fingertip distances, angles) to separate U/V/R and T/F
- [ ] Recording script with signer and session metadata
- [ ] Multi-signer dataset and signer-independent evaluation (leave-one-signer-out)
- [ ] Temporal modeling (sequence models such as GRU/LSTM) to recognize signs with movement and full words
- [ ] Web deployment so it can be tried without installing anything

## Tech stack

Python · OpenCV · MediaPipe · scikit-learn · NumPy · Matplotlib · pyttsx3