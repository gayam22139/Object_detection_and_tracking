
# Object Detection and Tracking

This repository contains an implementation of object detection and tracking using OpenCV. The project allows you to detect and track objects in video frames using various tracking algorithms.

## Features

- **Object Detection**: Detect objects in a video frame using a pre-trained cascade classifier.
- **Object Tracking**: Track detected objects across video frames using different tracking algorithms.
- **Supported Trackers**:
  - BOOSTING
  - MIL
  - KCF
  - TLD
  - MEDIANFLOW
  - MOSSE
  - CSRT

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Object_detection_and_tracking.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Object_detection_and_tracking
   ```

3. Create and activate a virtual environment (optional but recommended):

   ```bash
   python3 -m venv odt_env
   source odt_env/bin/activate
   ```

4. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Object Detector and Tracker

1. Place your input video in the project directory. A sample video (`input_video_1.mp4`) is included.
2. Run the `run.py` script to start the object detection and tracking process:

   ```bash
   python run.py
   ```

3. The program will prompt you to select a region of interest (ROI) in the video frame. You can select multiple ROIs by pressing any key (As of now, later this could be made such that detector will take care of it). Once you have selected the ROIs, press `Q` to start tracking.

### Testing the Detector

You can test the detector separately using the `test_detector.py` script:

```bash
python test_detector.py
```

## Customization

- **Tracker Selection**: You can choose different tracking algorithms by modifying the `tracker_type` variable in the `run.py` script.
- **Cascade Classifier**: You can use your own cascade classifier by replacing the files in the `cascade` directory. Use other pre-trained or custom object detctors this current version isn't detcting objects well try to use other detectors and improve it 

