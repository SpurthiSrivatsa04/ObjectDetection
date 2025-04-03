# Object Detection using YOLOv8 and OpenCV

This project implements object detection using the YOLOv8 model and OpenCV, enabling real-time object tracking in videos and webcam feeds.

## Features
- **YOLOv8-based Detection**: Uses a pre-trained YOLOv8 model for detecting objects in images, videos, and webcam streams.
- **Car Counting Module**: Tracks and counts vehicles in videos.
- **Multi-Object Tracking**: Implements SORT (Simple Online Realtime Tracker) for tracking detected objects.
- **Real-time Processing**: Supports live video and webcam-based object detection.

## Installation

### Prerequisites
Ensure you have Python 3.8+ installed.

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SpurthiSrivatsa04/ObjectDetection.git
   cd ObjectDetection
   ```
2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run Object Detection on Videos
```bash
python main.py --video videos/cars.mp4
```
- Replace `cars.mp4` with the desired video file.

### Run Real-Time Object Detection with Webcam
```bash
python yolowebcam/yolo-webcam.py
```

## Project Structure
```
ObjectDetection/
│── main.py                        # Entry point for object detection
│── requirements.txt               # Dependencies
│── carcounte/
│   ├── carcounter.py              # Vehicle counting module
│   ├── sort.py                    # SORT tracking algorithm
│── chapter5-runnin yolo/
│   ├── yolo basics.py             # YOLOv8 object detection implementation
│   ├── Images/                     # Example images
│── videos/
│   ├── bikes.mp4                   # Sample video files
│   ├── people.mp4
│── yolo weights/
│   ├── yolov8n.pt                 # Pre-trained YOLOv8 model weights
│── yolowebcam/
│   ├── yolo-webcam.py             # Webcam-based real-time detection
│── README.md                      # Documentation
```

## Contribution
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add new feature"`).
4. Push to GitHub (`git push origin feature-name`).
5. Open a pull request.

## License
This project is licensed under the MIT License.

