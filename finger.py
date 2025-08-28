import cv2
import mediapipe as mp
import pyttsx3

# Initialize speech engine
engine = pyttsx3.init()

# Mediapipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Webcam
cap = cv2.VideoCapture(0)

# Tips of fingers (thumb not included for simplicity)
finger_tips_ids = [8, 12, 16, 20]

prev_finger_count = -1

while True:
    success, image = cap.read()
    if not success:
        break

    # Flip image and convert to RGB
    image = cv2.flip(image, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    finger_count = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get landmarks
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = image.shape
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            # Count fingers
            if lm_list:
                # Thumb
                if lm_list[4][0] > lm_list[3][0]:  # Right hand assumption
                    finger_count += 1

                # Other 4 fingers
                for tip in finger_tips_ids:
                    if lm_list[tip][1] < lm_list[tip - 2][1]:
                        finger_count += 1

            # Draw landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show number on screen
    cv2.putText(image, f'Fingers: {finger_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 255, 0), 3)

    cv2.imshow("Finger Counter", image)

    # Speak only when the count changes
    if finger_count != prev_finger_count:
        engine.say(str(finger_count))
        engine.runAndWait()
        prev_finger_count = finger_count

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and cleanup
cap.release()
cv2.destroyAllWindows()
