import cv2
import mediapipe as mp
import numpy as np
import argparse
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video_dir', type=str, help='Path to directory with IPN Hand videos', required=True)
args = ap.parse_args()

vid_files = os.listdir(args.video_dir)
vid_paths = [os.path.join(args.video_dir, file) for file in vid_files]

# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

# For webcam input:
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('/born-again/datasets/IPN_Hand/videos/1CM1_1_R_#217.avi')
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
all_centers = np.zeros((length,2))
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            coords = np.zeros((21,2))
            for idx, landmark in enumerate(hand_landmarks.landmark):
              landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                   image.shape[1], image.shape[0])
              coords[idx,:] = landmark_px
            num_good = np.sum(~np.isnan(coords))
            coords = coords[~np.isnan(coords)].reshape((num_good // 2, 2))
            xmin = np.min(coords[:,0])
            xmax = np.max(coords[:,0])
            ymin = np.min(coords[:,1])
            ymax = np.max(coords[:,1])
            save_center = np.array([(xmin+xmax) / 2, (ymin+ymax) / 2])
            all_centers[idx,:] = save_center
            # center = (round((xmin+xmax) / 2), round((ymin+ymax) / 2))
            # cv2.circle(image, center, radius=2, color=(255,0,0), thickness=20)
            # cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(255,0,0), thickness=2)

            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break

np.save('')
cap.release()
