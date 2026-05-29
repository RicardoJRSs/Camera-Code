# Driver Attention Gaze Tracking System

This repository contains a Python and OpenFace-based gaze tracking system for measuring driver attention in a simulator. The system uses OpenFace to capture gaze data from a camera, then processes that data in Python to map gaze points to screen coordinates, detect Areas of Interest (AOIs), detect fixations, and generate CSV logs for analysis.

## Scripts

### `original.py`

This script is used for testing and visualization. It shows the calibration points, keeps the AOI overlay visible, displays the red gaze dot, and highlights the AOI currently being detected. Use this script to verify that the camera, OpenFace, calibration, gaze mapping, and logging are working correctly.

### `openface_pipeline.py`

This script is used for the actual simulator experiment. It shows the calibration screen first, then closes the visualization window and continues tracking in the background. This allows the simulator to run without the gaze visualization interfering with the display.

## Requirements

* Python 3.11 or compatible version
* OpenFace 2.2.0
* Camera/webcam
* `numpy`
* `keyboard`

Install the Python dependencies with:

```bash
pip install numpy keyboard
```

OpenFace must be installed separately. This repository does not include OpenFace or its model files.

## OpenFace Setup

Update the OpenFace path in the Python files before running:

```python
OPENFACE_EXE = r"C:\Users\rjriv\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
```

This path must point to the `FeatureExtraction.exe` file on your computer.

The scripts also use this folder for the live CSV output:

```python
OUTPUT_DIR = r"C:\OpenFaceLive"
```

Create the folder if it does not exist:

```bash
mkdir C:\OpenFaceLive
```

## Running the Scripts

To run the visual testing script:

```bash
python original.py
```

Press `Esc` to stop it.

To run the background experiment script:

```bash
python openface_pipeline.py
```

After calibration finishes, press `q` to stop the experiment.

## Output Files

Each run creates or overwrites these CSV files:

```text
aoi_visit_log.csv
fixation_log.csv
fixation_summary.csv
```

Save these files after each experiment before running the script again, because new runs overwrite previous results.

## Notes

For the three-monitor simulator setup, the system uses a 5760 × 1080 coordinate space based on three 1920 × 1080 monitors. If using a different screen setup, update the screen resolution values in the code.

OpenFace is an external dependency. Users must download and install OpenFace separately and follow OpenFace’s own license terms.

https://github.com/TadasBaltrusaitis/OpenFace

## License

This project is licensed under the MIT License. This license applies only to the code and documentation in this repository.
