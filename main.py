import cv2
import mediapipe as mp
import joblib
import numpy as np
import time
import sys
import subprocess
import threading
import queue

from features import extract_features


# =========================
# AUDIO
# =========================

speech_queue = queue.Queue()  # used to store words that need to be spoken
audio_lock = threading.Lock()  # prevents overlapping audio playback

last_spoken_word = ""
last_audio_time = 0
audio_cooldown = 0.1  # avoids speaking too frequently
SPEECH_RATE = 210  # words per minute for the macOS voice (default is ~175)
mac_synth = None  # macOS speech synthesizer, created once in the audio thread


def audio_worker():
    """
    Processes the queue of text that needs to be converted into speech. This function runs in a separate thread
    to avoid blocking the camera while the audio is being generated or played
    :return:
    None
    """
    global last_audio_time, mac_synth

    # macOS: create the speech synthesizer ONCE and keep it loaded in memory, so each word
    # starts almost instantly (calling the "say" command spawns a new process and takes ~2 s every time)
    if sys.platform == "darwin":
        try:
            from AppKit import NSSpeechSynthesizer
            mac_synth = NSSpeechSynthesizer.alloc().initWithVoice_(None)  # system default voice
            mac_synth.setRate_(SPEECH_RATE)

            # warm-up: the first word takes ~1.5 s because the speech engine initializes,
            # so we speak a silent word at startup to absorb that delay
            mac_synth.setVolume_(0.0)
            mac_synth.startSpeakingString_("ready")
            while mac_synth.isSpeaking():
                time.sleep(0.02)
            mac_synth.setVolume_(1.0)
        except ImportError:
            print("pyobjc not installed, falling back to the slower 'say' command")
            mac_synth = None

    while True:
        text = speech_queue.get()  # waits until something is added to the queue

        if text is None:
            break  # used to safely stop the thread

        try:
            speak_offline(text)
            last_audio_time = time.time()
        except Exception as e:
            print("Audio error:", e)

        speech_queue.task_done()


def speak_offline(text):
    """
    Speaks the given text using the operating system's offline voice. On macOS it uses the built-in "say" command,
    on Windows and Linux it uses pyttsx3. No internet connection is needed. The call blocks until the audio
    finishes, which is why it runs in the audio worker thread
    :param
    text: str
    Word or phrase that will be converted into speech
    :return:
    None
    """
    with audio_lock:  # ensures only one audio plays at a time
        if sys.platform == "darwin" and mac_synth is not None:
            mac_synth.startSpeakingString_(text)
            while mac_synth.isSpeaking():  # wait until the word finishes
                time.sleep(0.02)
        elif sys.platform == "darwin":
            subprocess.run(["say", "-r", str(SPEECH_RATE), text], check=False)
        else:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()


def speak(text):
    """
    Adds a word or phrase to the speech queue so it can be processed by the audio worker thread
    :param
    text: str
    Word or phrase to be spoken
    :return:
    None
    """
    clean_text = text.strip().lower()

    if clean_text:
        speech_queue.put(clean_text)  # enqueue instead of playing directly


audio_thread = threading.Thread(target=audio_worker, daemon=True)
audio_thread.start()


# =========================
# MODEL
# =========================

model = joblib.load("model/modelo_senas.pkl")  # trained classifier

# Safety check: the model must have been trained with the same features that extract_features produces.
# If the model is older (63 features), it would receive data it was not trained on and fail or predict garbage
EXPECTED_FEATURES = len(extract_features(np.zeros(63)))
if model.n_features_in_ != EXPECTED_FEATURES:
    sys.exit(
        f"The model expects {model.n_features_in_} features but features.py produces {EXPECTED_FEATURES}. "
        f"Retrain it with: python training.py"
    )


# =========================
# MEDIAPIPE
# =========================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,  # optimized for video
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# =========================
# VARIABLES
# =========================

current_prediction = ""
prediction_start_time = 0
text_output = ""

letter_confirmation_time = 0.5  # time needed to accept a letter
no_hand_space_time = 0.3  # time without hand to insert space (lower = faster audio, but too low may split words)

letter_cooldown = False
letter_cooldown_time = 0.8
last_letter_saved_time = 0

last_hand_seen_time = time.time()

is_translating = True
WINDOW_NAME = "Sign Translator"


# =========================
# MAIN LOOP
# =========================

def run_translator():
    """
    Runs the main loop of the application. It captures frames from the camera, detects hand landmarks using mediapipe,
    extracts the features (normalized coordinates + geometric features) and predicts the corresponding sign using the
    trained model. It also manages the logic for confirming letters, adding spaces and triggering audio output
    :return:
    None
    """
    global current_prediction, prediction_start_time, text_output
    global letter_cooldown, last_letter_saved_time, last_hand_seen_time
    global last_spoken_word, last_audio_time, is_translating

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera error")
        return

    cv2.namedWindow(WINDOW_NAME)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()

        if is_translating:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                last_hand_seen_time = now
                hand = results.multi_hand_landmarks[0]

                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                # flatten 21 landmarks → 63 raw values
                pts = []
                for lm in hand.landmark:
                    pts.extend([lm.x, lm.y, lm.z])

                # same feature extraction as in training (shared function in features.py),
                # reshaped to (1, n_features) because the model expects batch input
                features = extract_features(np.array(pts)).reshape(1, -1)
                pred = model.predict(features)[0]

                # CONFIRMATION LOGIC:
                # prevents noise by requiring stable prediction for some time
                if pred == current_prediction:
                    if now - prediction_start_time >= letter_confirmation_time:
                        if not letter_cooldown:
                            text_output += pred
                            letter_cooldown = True
                            last_letter_saved_time = now
                else:
                    current_prediction = pred
                    prediction_start_time = now
                    letter_cooldown = False

                # prevents adding same letter repeatedly
                if letter_cooldown and (now - last_letter_saved_time > letter_cooldown_time):
                    letter_cooldown = False

            else:
                # if no hand → interpret as space between words
                if now - last_hand_seen_time > no_hand_space_time:
                    if len(text_output) > 0 and text_output[-1] != " ":
                        text_output += " "

                        last_word = text_output.strip().split(" ")[-1]

                        # only speak new words and respect cooldown
                        if (
                            last_word
                            and last_word != last_spoken_word
                            and now - last_audio_time > audio_cooldown
                        ):
                            speak(last_word)
                            last_spoken_word = last_word

                    last_hand_seen_time = now

        cv2.putText(frame, f"Text: {text_output}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        # real-time control
        if key == ord("t"):
            is_translating = True
        elif key == ord("s"):
            is_translating = False
        elif key == ord("q"):
            break

    speech_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    run_translator()