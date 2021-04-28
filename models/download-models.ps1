echo "downloading models (this may take a while)..."
$ProgressPreference = 'SilentlyContinue'

wget https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands.cfg -O cross-hands.cfg
wget https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands.weights -O cross-hands.weights

wget https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-tiny.cfg -O cross-hands-tiny.cfg
wget https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-tiny.weights -O cross-hands-tiny.weights

wget https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-tiny-prn.cfg -O cross-hands-tiny-prn.cfg
wget https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-tiny-prn.weights -O cross-hands-tiny-prn.weights

wget https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-yolov4-tiny.cfg -O cross-hands-yolov4-tiny.cfg
wget https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-yolov4-tiny.weights -O cross-hands-yolov4-tiny.weights

echo "done!"