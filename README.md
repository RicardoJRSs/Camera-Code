# Driver Attention Gaze Tracking System

This repository contains a Python and OpenFace-based gaze tracking pipeline designed to measure driver attention in a driving simulator. The system captures webcam data through OpenFace, reads gaze information from a live CSV file, calibrates gaze angles to simulator screen coordinates, classifies gaze points into Areas of Interest (AOIs), detects fixations, and logs attention metrics for later analysis.

The project was tested in manual and autonomous driving simulator scenarios to compare gaze behavior across different driving conditions.

## Project Overview

The system follows a pipeline architecture:

```text
Camera → OpenFace → Live CSV File → Python Reader → Calibration → Gaze Mapping → AOI Detection → Fixation Detection → CSV Logs
```

OpenFace is used to extract facial, pose, and gaze features from webcam input. The Python scripts then process OpenFace’s output, specifically `gaze_angle_x` and `gaze_angle_y`, to estimate where the driver is looking on the simulator display.

The simulator display is treated as one continuous screen divided into AOIs:

* `LEFT`
* `CENTER_ROAD`
* `CENTER_DASHBOARD`
* `RIGHT`

The system outputs three CSV files:

* `aoi_visit_log.csv`
* `fixation_log.csv`
* `fixation_summary.csv`

These files can be used to analyze where the driver looked, how long they stayed in each AOI, how often fixations occurred, and how gaze shifted between regions.

## Repository Scripts

This repository includes two main Python scripts.

### `openface_visual_test.py`

This script is used for testing and visualization.

It keeps the AOI overlay and red gaze dot visible after calibration. This allows the user to verify that OpenFace, calibration, gaze mapping, AOI detection, and fixation detection are working correctly.

Use this script before running simulator experiments.

### `openface_pipeline.py`

This script is used for simulator experiments.

It shows the calibration screen first. After calibration is complete, the Tkinter visualization window closes and the system continues tracking in the background. This prevents the overlay from interfering with the simulator display.

Use this script when collecting experimental data.

## Requirements

### Hardware

* Integrated or external camera
* Computer with enough storage space
* Driving simulator setup or personal computer
* Three-monitor simulator setup recommended for full experiment mode

### Software

* Python 3.11 or compatible version
* OpenFace 2.2.0
* Visual Studio Code or another code editor
* CSV viewer such as Excel, Google Sheets, or LibreOffice Calc

### Python Libraries

Install the required Python libraries:

```bash
pip install numpy keyboard
```

The project also uses standard Python libraries:

```python
subprocess
time
os
csv
threading
tkinter
math
```

`tkinter` is usually included with Python on Windows.

## OpenFace Setup

Download and install OpenFace from the official repository:

```text
https://github.com/TadasBaltrusaitis/OpenFace
```

After installation, make sure the OpenFace folder contains:

```text
FeatureExtraction.exe
```

To test OpenFace on Windows, open PowerShell or Command Prompt inside the OpenFace folder and run:

```powershell
.\FeatureExtraction.exe -device 0 -gaze -pose -2Dfp -3Dfp
```

The value `0` represents the camera device. If the camera does not open, try:

```powershell
.\FeatureExtraction.exe -device 1 -gaze -pose -2Dfp -3Dfp
```

or:

```powershell
.\FeatureExtraction.exe -device 2 -gaze -pose -2Dfp -3Dfp
```

If OpenFace gives an error about missing CEN packages or missing models, run:

```powershell
.\download_models.ps1
```

If PowerShell blocks the script, temporarily allow the execution policy with:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then run:

```powershell
.\download_models.ps1
```

## Project Configuration

Before running the scripts, update the paths at the top of the Python file:

```python
OPENFACE_EXE = r"C:\Users\rjriv\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
OUTPUT_DIR = r"C:\OpenFaceLive"
OUTPUT_NAME = "live"
CSV_PATH = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}.csv")
```

`OPENFACE_EXE` must point to the exact location of `FeatureExtraction.exe` on your computer.

Create the OpenFace output folder:

```powershell
mkdir C:\OpenFaceLive
```

The live CSV file will be written to:

```text
C:\OpenFaceLive\live.csv
```

## Screen Resolution Setup

The scripts use screen coordinates to classify gaze points into AOIs.

For the three-monitor simulator setup, where each monitor is 1920×1080, use:

```python
SIM_MONITOR_WIDTH = 1920
SIM_MONITOR_HEIGHT = 1080
SIM_MONITOR_COUNT = 3

SCREEN_WIDTH_PX = SIM_MONITOR_WIDTH * SIM_MONITOR_COUNT
SCREEN_HEIGHT_PX = SIM_MONITOR_HEIGHT
```

This creates a full simulator coordinate system of:

```text
5760 × 1080 pixels
```

For a single-monitor setup, use the monitor’s actual resolution instead. For example:

```python
SCREEN_WIDTH_PX = 2560
SCREEN_HEIGHT_PX = 1600
```

In the `draw_loop()` function, the following lines can be used to automatically detect the current screen size:

```python
SCREEN_WIDTH_PX = root.winfo_screenwidth()
SCREEN_HEIGHT_PX = root.winfo_screenheight()
```

For the fixed three-monitor simulator setup, keep these lines commented. For single-monitor or personal-computer testing, uncomment them if automatic resolution detection is desired.

## Running the Visual Test Script

Use this script to test the system visually:

```bash
python openface_visual_test.py
```

This script will:

1. Start OpenFace.
2. Wait for the live CSV file.
3. Show calibration points.
4. Build the AOI regions.
5. Display the red gaze dot after calibration.
6. Highlight the AOI currently detected.
7. Save fixation and AOI visit logs when closed.

Press `Esc` to stop the script.

## Running the Background Experiment Script

Use this script for simulator experiments:

```bash
python openface_pipeline.py
```

This script will:

1. Start OpenFace.
2. Wait for the live CSV file.
3. Show calibration points.
4. Complete calibration.
5. Close the calibration window.
6. Continue tracking in the background.
7. Save fixation and AOI visit logs when stopped.

After calibration finishes, press:

```text
q
```

to stop the background experiment.

If the `q` key does not stop the experiment, run the terminal or Visual Studio Code as administrator. The `keyboard` library may require administrator permissions to detect global key presses.

## Calibration

During calibration, the user must look at each calibration point shown on the screen. The system collects:

```python
SAMPLES_PER_POINT = 75
```

This means the system collects 75 valid samples per calibration point. A valid sample must meet these conditions:

```python
success == 1
confidence >= 0.80
```

The calibration data is used by `fit_mapping()` to calculate the coefficient vectors `A_x` and `A_y`. These vectors are then used by `apply_mapping()` to convert new OpenFace gaze angles into simulator screen coordinates.

The mapping model is based on linear least-squares regression:

```text
sx = a(gx) + b(gy) + c
sy = d(gx) + e(gy) + f
```

Where:

* `gx` = `gaze_angle_x`
* `gy` = `gaze_angle_y`
* `sx` = estimated screen x-coordinate
* `sy` = estimated screen y-coordinate

## Output Files

Each run creates or overwrites the following CSV files:

### `aoi_visit_log.csv`

Records each AOI visit:

```text
aoi, start_time, end_time, duration_seconds
```

### `fixation_log.csv`

Records each detected fixation:

```text
fixation_start_time, fixation_end_time, duration_seconds, center_x, center_y, aoi
```

### `fixation_summary.csv`

Summarizes fixation results by AOI:

```text
aoi, fixation_count, total_fixation_duration, mean_fixation_duration, time_to_first_fixation
```

Important: each run overwrites the previous CSV results. Save or copy the output files after each experiment before running the script again.

## Common Problems and Solutions

### OpenFace does not detect the face

Improve lighting, adjust the camera angle, and make sure only one face is visible.

### OpenFace shows missing CEN packages or missing models

Run:

```powershell
.\download_models.ps1
```

inside the OpenFace folder.

### The script cannot find `FeatureExtraction.exe`

Update:

```python
OPENFACE_EXE
```

with the correct path.

### `live.csv` is not created

Make sure the output folder exists:

```text
C:\OpenFaceLive
```

Also verify that OpenFace has permission to write to that folder.

### AOIs do not align with the simulator

Check that the screen resolution values match the actual monitor setup.

For the three-monitor simulator setup, the expected coordinate system is:

```text
5760 × 1080
```

### The CSV result files are empty

Check whether OpenFace is producing valid frames:

```text
success = 1
confidence >= 0.80
```

Poor lighting, glasses, camera angle, or face obstruction can reduce confidence.

### The `q` key does not stop the background script

Run the terminal or Visual Studio Code as administrator.

## Recommended Workflow

1. Set the correct OpenFace path.
2. Set the correct screen resolution.
3. Run `openface_visual_test.py`.
4. Verify that the red dot and AOI highlighting work correctly.
5. Run `openface_pipeline.py` for the actual simulator experiment.
6. Stop the experiment with `q`.
7. Save the output CSV files in a separate results folder.

## License

This project is licensed under the MIT License. This license applies to the code in this repository. OpenFace is an external dependency and is distributed under its own license.
