# Sign Language Translator

Real-time sign language alphabet translator using a webcam. It detects the hand with **MediaPipe**, classifies the sign with a **Random Forest**, builds words letter by letter, and reads each finished word out loud.

<!-- TODO: add a GIF or screenshot of the app running -->
<!-- ![demo](demo.gif) -->

## Features

- Real-time recognition of the 26 letters of the <!-- TODO: ASL / LSEC / ... --> alphabet
- Letters are confirmed only after the prediction stays stable for 0.5 s, to filter out noise
- Removing your hand from the frame for 0.5 s inserts a space and ends the word
- Each finished word is spoken out loud (text-to-speech with `edge-tts`), in a separate thread so the camera never freezes
- Start screen built with `customtkinter`

## How it works

```
Webcam frame → MediaPipe Hands (21 landmarks × x,y,z) → normalization → Random Forest → letter
```

1. **Landmark extraction**: MediaPipe detects 21 hand landmarks, each with (x, y, z) coordinates, giving 63 features.
2. **Normalization**: coordinates are moved so the wrist is the origin, then scaled by the largest absolute value. This makes the prediction independent of where the hand is in the frame and how far it is from the camera.
3. **Classification**: a Random Forest with 500 trees predicts the letter.
4. **Text logic**: stable predictions become letters, hand absence becomes spaces, and finished words are sent to the speech queue.

## Results

- 26 classes (A–Z), ~300 samples per letter
- **~92% OOB accuracy**

## Project structure

```
├── main.py               # The translator app (camera, prediction, text and audio)
├── training.py           # Trains the Random Forest from the landmark dataset
├── coordinates.py        # Debug tool: shows the hand landmarks and their coordinates live
├── model/
│   └── modelo_senas.pkl  # Trained model
├── dataset_landmarks/    # Dataset: one folder per letter with .npy landmark files
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

Click **Start Camera** and start signing.

| Key | Action |
|-----|--------|
| `t` | Resume translating |
| `s` | Pause translating |
| `q` | Quit |

Tips:
- Hold each letter steady for about half a second until it appears.
- To finish a word, take your hand out of the frame. The word will be read out loud.
- Text-to-speech uses Microsoft's online voices through `edge-tts`, so it needs an internet connection. Recognition itself works offline.
- On macOS, allow camera access for your terminal or IDE the first time.

## Retraining the model

The dataset is included in `dataset_landmarks/`. To retrain:

```bash
python training.py
```

This normalizes every sample, trains the Random Forest, prints the OOB accuracy and saves the model to `model/modelo_senas.pkl`.

## Limitations

- Only static letters: signs with movement (such as J and Z) are learned as a single hand pose.
- One hand at a time.
- The dataset was recorded by one person, so accuracy may drop with other hands, lighting or camera angles.

## Roadmap

- [ ] Rigorous evaluation: held-out test set, per-letter metrics and confusion matrix
- [ ] Temporal modeling (sequence models such as LSTM) to recognize signs with movement and full words
- [ ] Web deployment so it can be tried without installing anything

## Tech stack

Python · OpenCV · MediaPipe · scikit-learn · NumPy · edge-tts · pygame · customtkinter