# AI-Based Concentration Tracker Using Computer Vision

A real-time concentration tracking application built using Python, OpenCV, MediaPipe, and Streamlit. The system analyzes facial and eye movements through a webcam to estimate visual attention and generate session-wise analytics.

## 1. Project Overview

The AI-Based Concentration Tracker is designed to monitor a user's visual attention during computer-based activities using webcam input.

The application uses facial landmarks and eye movement analysis to extract visual features such as Eye Aspect Ratio (EAR), gaze direction, blinking behaviour, and head pose.

These features are processed to calculate a visual attention score, helping users understand their attention patterns during a session.

The application also records session data and provides analytics through an interactive Streamlit dashboard.

## 2. Features

- **Real-Time Webcam Monitoring:** Captures live video through the webcam.
- **Face Mesh Detection:** Uses MediaPipe Face Mesh to detect facial landmarks.
- **Eye Aspect Ratio (EAR):** Analyzes eye openness and closure.
- **Blink Detection:** Detects and counts eye blinks.
- **Gaze Tracking:** Monitors horizontal eye movement.
- **Head Pose Estimation:** Estimates head orientation and movement.
- **Visual Attention Scoring:** Calculates an attention score using eye, gaze, and head movement measurements.
- **Calibration:** Establishes a baseline using initial facial measurements.
- **Session Recording:** Saves session measurements in CSV format.
- **Session Analytics:** Displays previously recorded session data.
- **CSV Export:** Allows users to download session records.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Interactive web dashboard |
| OpenCV | Webcam access and image processing |
| MediaPipe | Facial landmark detection |
| NumPy | Numerical calculations |
| Pandas | Data processing and analytics |
| Streamlit-WebRTC | Real-time webcam streaming |
| PyAV | Video frame processing |
| Protobuf | MediaPipe compatibility |

## 4. Project Structure

```text
concentration_tracker_DL/
│
├── app.py
├── requirements.txt
├── README.md
│
├── core/
│   ├── __init__.py
│   ├── camera.py
│   └── feature_extraction.py
│
├── data/
│   └── sessions/
│       └── session_*.csv
│
└── models/
```

### Directory Description

- `app.py`: Main Streamlit application.
- `requirements.txt`: Python package dependencies.
- `core/camera.py`: Webcam processing, facial landmark tracking, attention scoring, and session recording.
- `core/feature_extraction.py`: Eye, gaze, and head pose feature extraction.
- `data/sessions/`: Stores recorded session CSV files.
- `models/`: Reserved for future model development.

## 5. System Requirements

- Windows 10 or Windows 11
- Python 3.10
- Working webcam
- Internet connection for initial package installation


## 6. Installation and Setup

### Step 1: Clone or Download the Project

Download the project from GitHub or extract the project ZIP file.

Open Anaconda Prompt or Command Prompt.

Navigate to the project directory:

```bash
cd C:\concentration_tracker_DL
```

### Step 2: Create a Conda Environment

Create a dedicated Python environment:

```bash
conda create -n concentration_dl python=3.10 -y
```

Activate the environment:

```bash
conda activate concentration_dl
```

### Step 3: Install Dependencies

Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

The project uses Protobuf 4.25.8 for compatibility with the configured MediaPipe environment.

### Step 4: Verify the Installation

Run:

```bash
python -m pip check
```

You can also verify important imports:

```bash
python -c "import streamlit, cv2, mediapipe, numpy, pandas, av; print('All imports successful')"
```

## 7. How to Run the Application

Make sure the Conda environment is activated:

```bash
conda activate concentration_dl
```

Navigate to the project directory:

```bash
cd C:\concentration_tracker_DL
```

Start the Streamlit application:

```bash
streamlit run app.py
```

### Using the Application

1. Open the Concentration Tracker dashboard.
2. Allow webcam permission when prompted.
3. Start the camera.
4. Sit in front of the webcam and complete the calibration process.
5. Allow the system to establish your baseline facial measurements.
6. Begin the monitoring session.
7. Observe the live EAR, gaze, head pose, blink count, and visual attention score.
8. Stop the camera when the session is complete.
9. Open Session Analytics to view recorded session data.
10. Select and download a session CSV file.

## 8. How the Attention Scoring Works

The system estimates visual attention using three primary parameters:

1. **Eye Behaviour:** Uses EAR