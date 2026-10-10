import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                x_coords = []
                y_coords = []

                print("Coordenadas de la mano:")

                for i, landmark in enumerate(hand_landmarks.landmark):
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    z = landmark.z  # 👈 profundidad (no escalar)

                    x_coords.append(x)
                    y_coords.append(y)

                    print(f"Punto {i}: x={x}, y={y}, z={z:.4f}")

                print("-" * 40)

                # Dibujar mano
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Bounding box
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                padding = 20

                cv2.rectangle(
                    frame,
                    (x_min - padding, y_min - padding),
                    (x_max + padding, y_max + padding),
                    (0, 255, 0),
                    2
                )

        cv2.imshow("Detector de mano", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()