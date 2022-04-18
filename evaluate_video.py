import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
from kalman.kalman import KalmanFilter
from kalman.given_kalman import KalmanFilter as KF2
from yolo import YOLO

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video_path', type=str, help='Path to video', required=True)
args = ap.parse_args()

x_init = np.zeros(6)
P_init = np.diag(np.full(6, 500))
R_init = np.array([[9,0],[0,9]])
kf = KalmanFilter(1, x_init, P_init, R_init, 0.2**2, gain=1.)
first_pred = False

yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])

# For webcam input:
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(args.video_path)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
kfs = np.zeros((length,2))
yolos = np.zeros((length,2))
gt = np.zeros((length,2))

cnt = 0

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():

        success, frame = cap.read()
        if not success:
            break

        # YOLO INFERENCE
        width, height, inference_time, results = yolo.inference(frame)
        # display fps
        cv2.putText(frame, f'{round(1/inference_time,2)} FPS', (15,15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,255), 2)
        # sort by confidence
        results.sort(key=lambda x: x[2])
        # how many hands should be shown
        hand_count = len(results)
        # DETECTIONS
        if hand_count > 0:
            for detection in results[:hand_count]:
                first_pred = True
                id, name, confidence, x, y, w, h = detection
                if confidence < 0.3:
                    yolos[cnt,:] = [cx, cy]
                    frame = cv2.circle(frame, (round(cx), round(cy)), radius=2, color=(255,0,0), thickness=2)
                    kf.run(None)
                    xhat, yhat = kf.x[0], kf.x[3]
                    frame = cv2.circle(frame, (round(xhat), round(yhat)), radius=10, color=(0,255,0), thickness=20)
                    continue
                cx = x + (w / 2)
                cy = y + (h / 2)

                yolos[cnt,:] = [cx, cy]
                frame = cv2.circle(frame, (round(cx), round(cy)), radius=2, color=(255,0,0), thickness=2)

                z = np.array([cx, cy])
                kf.run(z)
                xhat, yhat = kf.x[0], kf.x[3]
                kfs[cnt,:] = [xhat, yhat]
                frame = cv2.circle(frame, (round(xhat), round(yhat)), radius=2, color=(0,255,0), thickness=2)
        elif first_pred:
            yolos[cnt,:] = [cx, cy]
            frame = cv2.circle(frame, (round(cx), round(cy)), radius=2, color=(255,0,0), thickness=2)
            kf.run(None)
            xhat, yhat = kf.x[0], kf.x[3]
            kfs[cnt,:] = [xhat, yhat]
            frame = cv2.circle(frame, (round(xhat), round(yhat)), radius=2, color=(0,255,0), thickness=2)

        # To improve performance, optionally mark the frame as not writeable to
        # pass by reference.
        # frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)

        # Draw the hand annotations on the frame.
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                coords = np.zeros((21,2))
                for idx, landmark in enumerate(hand_landmarks.landmark):
                  landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                       frame.shape[1], frame.shape[0])
                  coords[idx,:] = landmark_px

                num_good = np.sum(~np.isnan(coords))
                coords = coords[~np.isnan(coords)].reshape((num_good // 2, 2))
                xmin = np.min(coords[:,0])
                xmax = np.max(coords[:,0])
                ymin = np.min(coords[:,1])
                ymax = np.max(coords[:,1])
                gt_center = np.array([(xmin+xmax) / 2, (ymin+ymax) / 2])
                gt[cnt,:] = gt_center
                center = (round((xmin+xmax) / 2), round((ymin+ymax) / 2))
                cv2.circle(frame, center, radius=2, color=(0,0,255), thickness=2)
                # cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(255,0,0), thickness=2)

        # Flip the frame horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', frame)
        if cv2.waitKey(5) & 0xFF == 27:
          break

        cnt += 1

cap.release()
