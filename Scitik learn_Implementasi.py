import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('C:\\SEMESTER6\\KONTROL CERDAS\\PYCHARM\\model.p', 'rb'))
model=model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

labels_dict = {'0': 'buka', '1': 'tutup'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    if not ret or frame is None:
        print("Kamera tidak terbaca!")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            x_min, x_max = min(x_), max(x_)
            y_min, y_max = min(y_), max(y_)

            for landmark in hand_landmarks.landmark:
                data_aux.append((landmark.x - x_min) / (x_max - x_min))
                data_aux.append((landmark.y - y_min) / (y_max - y_min))

            def euclidean(p1, p2):
                return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
            finger_tips = [4, 8, 12, 16, 20]
            wrist = hand_landmarks.landmark[0]

        prediction = model.predict([np.array(data_aux)])
        predicted_character = labels_dict[str(prediction[0])]

        cv2.putText(frame,predicted_character,(200,100),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,0,0),3)
        cv2.imshow('frame',frame)
        cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()