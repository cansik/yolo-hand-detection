# Tracking Objects with Kalman Filter

## About

This project provides code for tracking objects that are detected in a detection system using the Kalman filter. The measurements from any black box deteciton system provide 2D coordinates in pixel space. The Kalman filter tracks noisy measurements from the detection system overtime to develop a linear model of motion using the hidden state of position, velocity, and acceleration in 2D.

We compare results to a different implementation of the Kalman filter (an OpenCV wrapper) from [PySource](https://pysource.com/).

In addition, we give credit to the theoretical implementation of the Kalman filter in this repository by Alex Becker, which can be found [here](https://www.kalmanfilter.net/multiExamples.html).

## Detection Systems

In this repository, we track hands (using YOLO) and tennis balls (using Haar features and AdaBoost). These detection systems are from the work of
- [cansik](https://github.com/cansik/yolo-hand-detection) (YOLO hand detection)
- [radosz99](https://github.com/radosz99/tennis-ball-detector) (Haar tennis ball detection)

## Requirements

Requirements can be installed using the following command:
```
pip3 install -r requirements.txt
```

## Running the Kalman filter for Hand Detection

The following command can be used to run the Kalman filter using YOLO hand detection:
```
python3 run_kalman.py -v /path/to/video.mp4 -c [confidence threshold]
```

## Notebooks

- A notebook with analysis of the Kalman filter in response to synthetic data can be found in `kalman/synthetic_data_kalman_tests.ipynb`
- A notebook with analysis of the Kalman filter in response to real data can be found in `kalman/tennis_ball_kalman_tests.ipynb`

## Videos and Results

Original videos and videos with the Kalman filter visualized on some real data can be found [here](https://drive.google.com/drive/folders/1_GMcCwXVuuLb_UNYZ_SyYXS-Ouidv4xr?usp=sharing) (a CU email address is needed for access).
