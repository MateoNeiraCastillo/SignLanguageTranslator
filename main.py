import cv2
import mediapipe as mp
import joblib
import numpy as np
import time
import os
import threading
import queue
import asyncio
import tempfile
import pygame
import edge_tts
import customtkinter as ctk


# =========================
# AUDIO
# =========================

pygame.mixer.init()

speech_queue = queue.Queue()  # used to store words that need to be spoken
audio_lock = threading.Lock()  # prevents overlapping audio playback

last_spoken_word = ""
last_audio_time = 0
audio_cooldown = 0.1  # avoids speaking too frequently

VOICE = "en-US-JennyNeural"


def audio_worker():
    """
    Processes the queue of text that needs to be converted into speech. This function runs in a separate thread
    to avoid blocking the camera while the audio is being generated or played
    :return:
    None
    """
    global last_audio_time

    while True:
        text = speech_queue.get()  # waits until something is added to the queue

        if text is None:
            break  # used to safely stop the thread

        try:
            speak_with_edge_tts(text)
            last_audio_time = time.time()
        except Exception as e:
            print("Audio error:", e)

        speech_queue.task_done()


def speak_with_edge_tts(text):
    """
    Generates an audio file from the given text using edge_tts and plays it using pygame. A temporary file is created
    and deleted after playback
    :param
    text: str
    Word or phrase that will be converted into speech
    :return:
    None
    """
    with audio_lock:  # ensures only one audio plays at a time
        fd, temp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)

        async def generate():
            communicate = edge_tts.Communicate(text=text, voice=VOICE)
            await communicate.save(temp_path)

        asyncio.run(generate())

        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()

        # wait until audio finishes (this would freeze program if not in a thread)
        while pygame.mixer.music.get_busy():
            time.sleep(0.03)

        pygame.mixer.music.unload()

        if os.path.exists(temp_path):
            os.remove(temp_path)  # clean temp file


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

model = joblib.load("model/modelo_senas.pkl")  # trained classifier


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
no_hand_space_time = 0.5  # time without hand to insert space

letter_cooldown = False
letter_cooldown_time = 0.8
last_letter_saved_time = 0

last_hand_seen_time = time.time()

is_translating = True
WINDOW_NAME = "Sign Translator"

# =========================
# START SCREEN
# =========================

def start_screen():
    """
    Displays the initial UI where the user can start the camera or close the application
    :return:
    bool
    True if the user wants to start the camera, False if the user closes the application
    """
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("green")

    app = ctk.CTk()
    app.title("Sign Language Translator Pro")
    app.attributes("-fullscreen", True)
    app.resizable(False, False)

    start_camera = {"value": False}

    frame = ctk.CTkFrame(app, corner_radius=28)
    frame.pack(expand=True, fill="both", padx=28, pady=28)

    title = ctk.CTkLabel(
        frame,
        text="SIGN TRANSLATOR",
        font=("Arial Rounded MT Bold", 34)
    )
    title.pack(pady=(42, 8))

    subtitle = ctk.CTkLabel(
        frame,
        text="Real-time sign language recognition",
        font=("Arial", 16)
    )
    subtitle.pack(pady=(0, 28))

    def launch():
        """
        Closes the start screen and allows the camera translator to start
        :return:
        None
        """
        start_camera["value"] = True
        app.destroy()

    # 🔥 NICE BUTTON (your original style)
    start_btn = ctk.CTkButton(
        frame,
        text="Start Camera",
        font=("Arial Rounded MT Bold", 18),
        height=52,
        width=240,
        corner_radius=18,
        command=launch
    )
    start_btn.pack(pady=8)

    # 🔥 SECOND BUTTON (styled)
    quit_btn = ctk.CTkButton(
        frame,
        text="Close",
        font=("Arial", 15),
        height=42,
        width=180,
        corner_radius=16,
        fg_color="#3A1E1E",
        hover_color="#612828",
        command=app.destroy
    )
    quit_btn.pack(pady=(14, 0))

    app.mainloop()

    return start_camera["value"]

# =========================
# MAIN LOOP
# =========================

def run_translator():
    """
    Runs the main loop of the application. It captures frames from the camera, detects hand landmarks using mediapipe,
    normalizes them and predicts the corresponding sign using the trained model. It also manages the logic for
    confirming letters, adding spaces and triggering audio output
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

                # flatten 21 landmarks → 63 features
                pts = []
                for lm in hand.landmark:
                    pts.extend([lm.x, lm.y, lm.z])

                # reshape to (1,63) because model expects batch input
                features = normalize(np.array(pts)).reshape(1, -1)
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
    pygame.quit()


if start_screen():
    run_translator()