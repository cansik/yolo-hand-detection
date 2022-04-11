import argparse
import cv2
from kalman.kalman import KalmanFilter

from yolo import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--network', default="normal", choices=["normal", "tiny", "prn", "v4-tiny"],
                help='Network Type')
ap.add_argument('-v', '--video_path', type=str, help='Path to video', required=True)
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
ap.add_argument('-nh', '--hands', default=-1, help='Total number of hands to be detected per frame (-1 for all)')
args = ap.parse_args()

if args.network == "normal":
    print("loading yolo...")
    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
elif args.network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
elif args.network == "v4-tiny":
    print("loading yolov4-tiny-prn...")
    yolo = YOLO("models/cross-hands-yolov4-tiny.cfg", "models/cross-hands-yolov4-tiny.weights", ["hand"])
else:
    print("loading yolo-tiny...")
    yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

cv2.namedWindow("preview")
cap = cv2.VideoCapture(args.video_path)

result = cv2.VideoWriter(f'after.mp4',
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        30.0, (1920, 1080))

x_init = np.zeros(6)
P_init = np.diag(np.full(6, 500))
R_init = np.array([[9,0],[0,9]])
kf = KalmanFilter(1, x_init, P_init, R_init, 0.2**2)

while cap.isOpened():

    ok, frame = cap.read()
    if not ok:
        break

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
    for detection in results[:hand_count]:
        id, name, confidence, x, y, w, h = detection
        cx = x + (w / 2)
        cy = y + (h / 2)

        # draw a bounding box rectangle and label on the image
        color = (0, 255, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = "%s (%s)" % (name, round(confidence, 2))
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

        frame = cv2.circle(frame, (round(cx), round(cy)), radius=10, color=(0,0,255), thickness=20)

    cv2.imshow("preview", frame)
    result.write(frame)

    # rval, frame = vc.read()

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
cap.release()
result.release()
