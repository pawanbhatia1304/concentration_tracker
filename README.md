# Concentration Tracker

A real-time concentration tracking system built using **MediaPipe** and **OpenCV**. This tool evaluates a user's attentiveness based on eye blinks, gaze direction, and head pose — ideal for applications like study monitoring, e-learning, or productivity enhancement.

## Features

- **Eye Blink Detection**  
  Calculates Eye Aspect Ratio (EAR) to detect blinks and periods of eye closure.

- **Gaze Detection**  
  Estimates if the user is looking straight or away using iris landmarks.

- **Head Pose Estimation**  
  Evaluates user orientation based on nose position relative to screen center.

- **Concentration Score**  
  Computes a weighted score combining gaze, head pose, and blinking behavior.

- **Live Visual Feedback**  
  Real-time UI overlay on webcam feed showing concentration level, blink status, and distraction counter.

- **Distraction Tracking**  
  Counts how many frames the user is not paying attention and issues warnings if needed.

## Sample Output

The video feed displays:
- A concentration percentage bar
- Blink detection alerts
- Distraction count
- ACTIVE / DISTRACTED indicator
- FPS counter

## Tech Stack

- Python 3.x
- OpenCV
- MediaPipe (FaceMesh)
- NumPy

## How It Works

1. **Face landmarks** are detected using MediaPipe FaceMesh.
2. **EAR (Eye Aspect Ratio)** is used to detect blinks.
3. **Iris position** is used to assess gaze direction.
4. **Nose position** is used to infer head pose.
5. A **composite concentration score** is calculated as: score = 0.4 * gaze + 0.4 * head_pose + 0.2 * (not blinking)
6. A **visual feedback system** shows user concentration in real time.

## Run the Project

> [!IMPORTANT]
> **Python Version Compatibility:** 
> MediaPipe's legacy solutions API (`mp.solutions`) is not supported on newer/prerelease Python versions like **Python 3.13 or 3.14**. To run this application, it is highly recommended to use **Python 3.11** or **Python 3.10**.

You can run this project locally in a terminal using one of the following methods:

### Option A: Using `uv` (Recommended - fastest & cleanest)

If you have `uv` installed, you can run the script directly with Python 3.11 without polluting your global environment:

```bash
# Clone the repository and enter the directory
git clone https://github.com/pawanbhatia1304/concentration_tracker
cd concentration_tracker

# Run the script directly using uv (which automatically sets up Python 3.11 and dependencies)
uv run --python 3.11 --with opencv-python --with numpy --with "mediapipe<0.10.30" python concentration_tracker.py
```

Alternatively, you can create a local virtual environment using `uv`:

```bash
# Create a Python 3.11 virtual environment
uv venv --python 3.11

# Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (CMD):
.venv\Scripts\activate.bat
# On macOS/Linux:
source .venv/bin/activate

# Install the required packages
uv pip install opencv-python numpy "mediapipe<0.10.30"

# Run the tracker
python concentration_tracker.py
```

---

### Option B: Using Conda

If you use Conda, you can set up a dedicated environment with Python 3.11:

```bash
# Create a new environment with Python 3.11
conda create -n tracker python=3.11 -y

# Activate the environment
conda activate tracker

# Install the required packages
pip install opencv-python numpy "mediapipe<0.10.30"

# Run the tracker
python concentration_tracker.py
```

---

### Option C: Using Standard Virtual Environment (venv)

If you already have Python 3.11 or 3.10 installed on your system as `python3.11` (or if it is your default interpreter), you can use the standard `venv` module:

```bash
# Create a virtual environment
python3.11 -m venv venv

# Activate the virtual environment
# On Windows (PowerShell):
venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run the tracker
python concentration_tracker.py
```
