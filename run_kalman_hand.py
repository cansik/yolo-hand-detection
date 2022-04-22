import argparse
import cv2
from kalman.kalman import KalmanFilter
from kalman.given_kalman import KalmanFilter as KalmanFilter2
import numpy as np

from yolo import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video_path', type=str, help='Path to video', required=True)
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
ap.add_argument('-nh', '--hands', default=-1, help='Total number of hands to be detected per frame (-1 for all)')
args = ap.parse_args()

yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])

yolo.size = int(args.size) # input shape resolution for YOLO (must be square)
yolo.confidence = float(args.confidence) # confidence threshold for YOLO detections

cv2.namedWindow("preview")
cap = cv2.VideoCapture(args.video_path)
np_path = args.video_path[:-4]+'.npy' # path to save detection as measurements to a numpy file
new_path = args.video_path[:-4]+'_results.mp4' # path to save results video
measurements = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 2)) # initialize saved measurements
fcnt = 0 # frame count

# write the results to a video with 30 FPS
result = cv2.VideoWriter(new_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        30.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# initialization of Kalman filter variables from https://www.kalmanfilter.net/multiExamples.html
x_init = np.zeros(6)
P_init = np.diag(np.full(6, 500))
R_init = np.array([[9,0],[0,9]])
kf = KalmanFilter(1, x_init, P_init, R_init, 0.2**2, gain=1.) # initialize the Kalman filter we DID write

kf2 = KalmanFilter2() # initialize a Kalman filter we DID NOT write for comparison in the report

first_pred = False # flag that stores whether we have made an initial detection on a hand

# while we have frames in input the video to read from
while cap.isOpened():

    ok, frame = cap.read()
    if not ok: # break out of this loop if we reach the end of the video
        break

    # YOLO inference from https://github.com/cansik/yolo-hand-detection
    width, height, inference_time, results = yolo.inference(frame)

    # display fps
    cv2.putText(frame, f'{round(1/inference_time,2)} FPS', (15,15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,255), 2)

    # sort by confidence
    results.sort(key=lambda x: x[2])

    # how many hands should be shown
    hand_count = len(results)
    if args.hands != -1:
        hand_count = int(args.hands)

    # display hands
    if hand_count > 0:
        for detection in results[:hand_count]:
            first_pred = True
            id, name, confidence, x, y, w, h = detection

            # get center of object as thing to track
            cx = x + (w / 2)
            cy = y + (h / 2)

            measurements[fcnt,:] = [cx, cy] # add the box center to measurements

            # draw a bounding box rectangle and label on the image
            color = (255, 255, 0) # teal
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "%s (%s)" % (name, round(confidence, 2))
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

            frame = cv2.circle(frame, (round(cx), round(cy)), radius=5, color=(0,0,255), thickness=2) # red

            z = np.array([cx, cy]) # measurement vector
            xhat, yhat = kf.run(z) # run a step of "our" Kalman filter
            # xhat, yhat = kf.x[0], kf.x[3] # get the position elements in our state vector
            frame = cv2.circle(frame, (round(xhat), round(yhat)), radius=5, color=(255,0,0), thickness=2) # blue

            xhat2, yhat2 = kf2.predict([cx, cy]) # run a step of "their" Kalman filter
            frame = cv2.circle(frame, (round(xhat2), round(yhat2)), radius=5, color=(0,255,255), thickness=2) # yellow

    elif first_pred:
        # run extrapolation if we don't have a detection measurement
        kf.predict()
        xhat, yhat = kf.x[0], kf.x[3]
        frame = cv2.circle(frame, (round(xhat), round(yhat)), radius=5, color=(255,0,0), thickness=2)

        predicted = kf2.kf.predict()
        xhat2, yhat2 = int(predicted[0]), int(predicted[1])
        frame = cv2.circle(frame, (round(xhat2), round(yhat2)), radius=5, color=(0,255,255), thickness=2)

    cv2.imshow("preview", frame)
    result.write(frame)
    fcnt += 1

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

np.save(np_path, measurements) # save the measurement for analysis in Jupyter
cv2.destroyWindow("preview")
cap.release() # video in
result.release() # video out
