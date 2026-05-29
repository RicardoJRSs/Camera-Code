import subprocess
import time
import os
import csv
import threading
import tkinter as tk
import numpy as np
import math
import keyboard

# =========================================================
# CHANGE THESE PATHS TO MATCH YOUR COMPUTER
# =========================================================
OPENFACE_EXE = r"C:\Users\rjriv\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
OUTPUT_DIR = r"C:\OpenFaceLive"
OUTPUT_NAME = "live"
CSV_PATH = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}.csv")

# =========================================================
# SCREEN SETTINGS
# These are updated automatically from tkinter at runtime.
# =========================================================

#Adjust these values to your screen setup. The default is set for a triple monitor setup with 1920x1080 monitors. If you are using a single monitor, you can comment out the SIM_MONITOR variables and just set SCREEN_WIDTH_PX and SCREEN_HEIGHT_PX to your monitor's resolution.
#There is also code in the draw_loop function that will automatically update SCREEN_WIDTH_PX and SCREEN_HEIGHT_PX to match the resolution of the primary monitor, which is helpful for multi-monitor setups. You can comment that out if you want to use fixed values instead.
SCREEN_WIDTH_PX = 2560
SCREEN_HEIGHT_PX = 1600

# SIM_MONITOR_WIDTH = 1920
# SIM_MONITOR_HEIGHT = 1080
# SIM_MONITOR_COUNT = 3

# SCREEN_WIDTH_PX = SIM_MONITOR_WIDTH * SIM_MONITOR_COUNT
# SCREEN_HEIGHT_PX = SIM_MONITOR_HEIGHT

# =========================================================
# PIPELINE SETTINGS
# =========================================================
CONFIDENCE_THRESHOLD = 0.80
SAMPLES_PER_POINT = 75
SMOOTHING_ALPHA = 0.15
TARGET_SETTLE_SECONDS = 0.8
MIN_AOI_DWELL_SECONDS = 0.20

# =========================================================
# FIXATION SETTINGS
# =========================================================
FIXATION_RADIUS_PX = 80
FIXATION_MIN_DURATION_SECONDS = 0.20

# =========================================================
# OUTPUT LOG FILES
# =========================================================
AOI_VISIT_LOG = "aoi_visit_log.csv"
FIXATION_LOG = "fixation_log.csv"
FIXATION_SUMMARY_LOG = "fixation_summary.csv"

# Shared gaze point for drawing
current_point = [SCREEN_WIDTH_PX // 2, SCREEN_HEIGHT_PX // 2]
status_text = "Starting..."
latest_angles = [None, None]

# Calibration / mapping globals
calibration_done = False
mapping_ready = False
A_x = None
A_y = None

# 5-point calibration targets
calibration_targets = []
current_target_index = 0
current_target_samples = []
last_target_switch_time = 0.0

# Synchronization
targets_ready = threading.Event()

# AOI globals
AOIS = {}
current_aoi = None
pending_aoi = None
pending_aoi_start = None
aoi_entry_time = None

# Fixation globals
fixation_active = False
fixation_start_time = None
fixation_center = None
fixation_points = []

stop_event = threading.Event()
calibration_finished_event = threading.Event()
experiment_start_time = None
state_lock = threading.Lock()

def init_logs():
    with open(AOI_VISIT_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["aoi", "start_time", "end_time", "duration_seconds"])

    with open(FIXATION_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fixation_start_time",
            "fixation_end_time",
            "duration_seconds",
            "center_x",
            "center_y",
            "aoi"
        ])

    with open(FIXATION_SUMMARY_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "aoi",
            "fixation_count",
            "total_fixation_duration",
            "mean_fixation_duration",
            "time_to_first_fixation"
        ])


def log_visit(aoi, start_time, end_time):
    duration = end_time - start_time
    with open(AOI_VISIT_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([aoi, start_time, end_time, duration])


def log_fixation(start_time, end_time, center_x, center_y, aoi):
    duration = end_time - start_time
    with open(FIXATION_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([start_time, end_time, duration, center_x, center_y, aoi])


def write_fixation_summary():
    global experiment_start_time

    if not os.path.exists(FIXATION_LOG):
        return

    with open(FIXATION_LOG, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return

    valid_rows = []
    for r in rows:
        aoi = r.get("aoi", "")
        if aoi and aoi != "None":
            valid_rows.append(r)

    with open(FIXATION_SUMMARY_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "aoi",
            "fixation_count",
            "total_fixation_duration",
            "mean_fixation_duration",
            "time_to_first_fixation"
        ])

        if not valid_rows:
            return

        if experiment_start_time is not None:
            experiment_start = experiment_start_time
        else:
            experiment_start = min(float(r["fixation_start_time"]) for r in valid_rows)

        summary = {}

        for r in valid_rows:
            aoi = r["aoi"]
            start_time = float(r["fixation_start_time"])
            duration = float(r["duration_seconds"])

            if aoi not in summary:
                summary[aoi] = {
                    "count": 0,
                    "total_duration": 0.0,
                    "first_fixation_start": start_time
                }

            summary[aoi]["count"] += 1
            summary[aoi]["total_duration"] += duration

            if start_time < summary[aoi]["first_fixation_start"]:
                summary[aoi]["first_fixation_start"] = start_time

        for aoi, vals in summary.items():
            fixation_count = vals["count"]
            total_fixation_duration = vals["total_duration"]
            mean_fixation_duration = (
                total_fixation_duration / fixation_count if fixation_count > 0 else 0.0
            )
            time_to_first_fixation = vals["first_fixation_start"] - experiment_start

            writer.writerow([
                aoi,
                fixation_count,
                total_fixation_duration,
                mean_fixation_duration,
                time_to_first_fixation
            ])


def clean_old_csv():
    if os.path.exists(CSV_PATH):
        try:
            os.remove(CSV_PATH)
            print("Old CSV deleted.")
        except Exception as e:
            print(f"Could not delete old CSV: {e}")


def start_openface():
    if not os.path.exists(OPENFACE_EXE):
        raise FileNotFoundError(f"OpenFace executable not found: {OPENFACE_EXE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)

    cmd = [
        OPENFACE_EXE,
        "-device", "0",
        "-of", CSV_PATH,
        "-2Dfp",
        "-3Dfp",
        "-pose",
        "-gaze",
    ]

    print("Starting OpenFace...")
    return subprocess.Popen(cmd, cwd=os.path.dirname(OPENFACE_EXE))

def wait_for_csv(timeout=15):
    print("Waiting for new CSV...")
    start = time.time()

    while True:
        if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:
            print(f"CSV detected: {CSV_PATH}")
            return

        if time.time() - start > timeout:
            raise TimeoutError("CSV was not created in time.")

        time.sleep(0.1)


def read_header(timeout=10):
    start = time.time()

    while True:
        try:
            with open(CSV_PATH, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline()
                if first_line:
                    header = next(csv.reader([first_line]))
                    header = [h.strip() for h in header]
                    return header
        except Exception:
            pass

        if time.time() - start > timeout:
            raise TimeoutError("Could not read CSV header in time.")

        time.sleep(0.1)


def parse_float(row, key, default=None):
    try:
        value = row.get(key, default)
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp_screen_point(pt):
    if pt is None:
        return None
    px, py = pt
    px = max(0, min(SCREEN_WIDTH_PX - 1, int(px)))
    py = max(0, min(SCREEN_HEIGHT_PX - 1, int(py)))
    return (px, py)


# def build_calibration_targets():
#     global calibration_targets

#     margin_x = int(SCREEN_WIDTH_PX * 0.25)
#     margin_y = int(SCREEN_HEIGHT_PX * 0.22)

#     cx = SCREEN_WIDTH_PX // 2
#     cy = SCREEN_HEIGHT_PX // 2

#     calibration_targets = [
#         ("CENTER",       (cx, cy)),
#         ("TOP_LEFT",     (margin_x, margin_y)),
#         ("TOP_RIGHT",    (SCREEN_WIDTH_PX - margin_x, margin_y)),
#         ("BOTTOM_LEFT",  (margin_x, SCREEN_HEIGHT_PX - margin_y)),
#         ("BOTTOM_RIGHT", (SCREEN_WIDTH_PX - margin_x, SCREEN_HEIGHT_PX - margin_y)),
#     ]
def build_calibration_targets():
    global calibration_targets

    third = SCREEN_WIDTH_PX // 3

    #This moves calibration targets slightly inward from the edges, which seems to help with OpenFace's accuracy on the left and right thirds of the screen.
    #Because the simulator is bigger than a typical laptop screen, the edges are more extreme and OpenFace has more trouble with them. Adjust as needed for your setup.
    #Also the rear view mirror in the simulator overlap with the center monitor so moving the calibration targets inward helps create more sensitivity around transitions between center to right and center to left.
    left_x = int(third * 0.75)
    center_x = third + (third // 2)
    right_x = int(2 * third + (third * 0.25))

    top_y = int(SCREEN_HEIGHT_PX * 0.22)
    road_y = int(SCREEN_HEIGHT_PX * 0.35)
    center_y = SCREEN_HEIGHT_PX // 2
    dash_y = int(SCREEN_HEIGHT_PX * 0.75)
    bottom_y = int(SCREEN_HEIGHT_PX * 0.82)

    calibration_targets = [
        ("LEFT_TOP", (left_x, top_y)),
        ("LEFT_CENTER", (left_x, center_y)),
        ("LEFT_BOTTOM", (left_x, bottom_y)),

        ("CENTER_ROAD", (center_x, road_y)),
        ("CENTER_CENTER", (center_x, center_y)),
        ("CENTER_DASHBOARD", (center_x, dash_y)),

        ("RIGHT_TOP", (right_x, top_y)),
        ("RIGHT_CENTER", (right_x, center_y)),
        ("RIGHT_BOTTOM", (right_x, bottom_y)),
    ]

def build_aois():
    global AOIS

    third = SCREEN_WIDTH_PX // 3
    half_height = SCREEN_HEIGHT_PX // 2

    AOIS = {
        "LEFT": (0, 0, third, SCREEN_HEIGHT_PX),

        "CENTER_ROAD": (
            third,
            0,
            2 * third,
            half_height
        ),

        "CENTER_DASHBOARD": (
            third,
            half_height,
            2 * third,
            SCREEN_HEIGHT_PX
        ),

        "RIGHT": (2 * third, 0, SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX),
    }


def fit_mapping(calib_pairs):
    """
    calib_pairs: list of ((gx, gy), (sx, sy))
    Fits:
        sx = a*gx + b*gy + c
        sy = d*gx + e*gy + f
    """
    G = []
    Sx = []
    Sy = []

    for (gx, gy), (sx, sy) in calib_pairs:
        G.append([gx, gy, 1.0])
        Sx.append(sx)
        Sy.append(sy)

    G = np.array(G, dtype=float)
    Sx = np.array(Sx, dtype=float)
    Sy = np.array(Sy, dtype=float)

    coef_x, *_ = np.linalg.lstsq(G, Sx, rcond=None)
    coef_y, *_ = np.linalg.lstsq(G, Sy, rcond=None)
    return coef_x, coef_y


def apply_mapping(gx, gy):
    global A_x, A_y
    if A_x is None or A_y is None:
        return None

    sx = A_x[0] * gx + A_x[1] * gy + A_x[2]
    sy = A_y[0] * gx + A_y[1] * gy + A_y[2]
    return clamp_screen_point((sx, sy))


def detect_aoi(x, y):
    for name, (x1, y1, x2, y2) in AOIS.items():
        if x1 <= x < x2 and y1 <= y < y2:
            return name
    return None


def update_aoi_state(x, y):
    global current_aoi, pending_aoi, pending_aoi_start, aoi_entry_time

    with state_lock:
        now = time.time()
        new_aoi = detect_aoi(x, y)

        if new_aoi == current_aoi:
            pending_aoi = None
            pending_aoi_start = None
            return

        if new_aoi != pending_aoi:
            pending_aoi = new_aoi
            pending_aoi_start = now
            return

        if pending_aoi_start is not None and (now - pending_aoi_start) >= MIN_AOI_DWELL_SECONDS:
            if current_aoi is not None and aoi_entry_time is not None:
                log_visit(current_aoi, aoi_entry_time, now)

            current_aoi = pending_aoi
            aoi_entry_time = now
            pending_aoi = None
            pending_aoi_start = None


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def update_fixation_state(x, y):
    global fixation_active, fixation_start_time, fixation_center, fixation_points

    with state_lock:
        now = time.time()
        pt = (x, y)

        if fixation_center is None:
            fixation_center = pt
            fixation_start_time = now
            fixation_points = [pt]
            fixation_active = False
            return

        if distance(pt, fixation_center) <= FIXATION_RADIUS_PX:
            fixation_points.append(pt)

            avg_x = sum(p[0] for p in fixation_points) / len(fixation_points)
            avg_y = sum(p[1] for p in fixation_points) / len(fixation_points)
            fixation_center = (avg_x, avg_y)

            if (not fixation_active) and (now - fixation_start_time >= FIXATION_MIN_DURATION_SECONDS):
                fixation_active = True

            return

        if fixation_active:
            end_time = now
            aoi = detect_aoi(int(fixation_center[0]), int(fixation_center[1]))
            log_fixation(
                fixation_start_time,
                end_time,
                int(fixation_center[0]),
                int(fixation_center[1]),
                aoi
            )

        fixation_center = pt
        fixation_start_time = now
        fixation_points = [pt]
        fixation_active = False

def finalize_fixation():
    global fixation_active, fixation_start_time, fixation_center, fixation_points

    with state_lock:
        if fixation_active and fixation_center is not None and fixation_start_time is not None:
            end_time = time.time()
            fx = int(fixation_center[0])
            fy = int(fixation_center[1])
            aoi = detect_aoi(fx, fy)
            log_fixation(fixation_start_time, end_time, fx, fy, aoi)

        fixation_active = False
        fixation_start_time = None
        fixation_center = None
        fixation_points = []


def read_csv_live(fieldnames):
    global latest_angles, status_text
    global current_target_index, current_target_samples, last_target_switch_time
    global calibration_done, mapping_ready, A_x, A_y
    global current_point, experiment_start_time

    while not targets_ready.is_set():
        if stop_event.is_set():
            return
        time.sleep(0.01)

    print("Reading OpenFace data live...")
    collected_pairs = []

    with open(CSV_PATH, "r", encoding="utf-8", errors="ignore") as f:
        f.readline()  # discard header

        while not stop_event.is_set():
            line = f.readline()

            if not line:
                time.sleep(0.01)
                continue

            try:
                values = next(csv.reader([line]))
                values = [v.strip() for v in values]
            except Exception:
                continue

            if len(values) != len(fieldnames):
                continue

            row = dict(zip(fieldnames, values))

            confidence = parse_float(row, "confidence")
            success = parse_float(row, "success")

            if success != 1:
                continue
            if confidence is None or confidence < CONFIDENCE_THRESHOLD:
                continue

            gaze_angle_x = parse_float(row, "gaze_angle_x")
            gaze_angle_y = parse_float(row, "gaze_angle_y")

            if gaze_angle_x is None or gaze_angle_y is None:
                continue

            with state_lock:
                latest_angles[0] = gaze_angle_x
                latest_angles[1] = gaze_angle_y

                local_calibration_done = calibration_done
                local_target_index = current_target_index
                local_last_target_switch_time = last_target_switch_time

            if not local_calibration_done:
                now = time.time()

                if local_target_index >= len(calibration_targets):
                    continue

                target_name, target_xy = calibration_targets[local_target_index]

                if now - local_last_target_switch_time < TARGET_SETTLE_SECONDS:
                    remaining = TARGET_SETTLE_SECONDS - (now - local_last_target_switch_time)
                    with state_lock:
                        status_text = f"Look at {target_name}... collecting in {remaining:.1f}s"
                    continue

                with state_lock:
                    status_text = f"Look at {target_name} ({len(current_target_samples)}/{SAMPLES_PER_POINT})"
                    current_target_samples.append((gaze_angle_x, gaze_angle_y))

                    if len(current_target_samples) < SAMPLES_PER_POINT:
                        continue

                    samples_copy = current_target_samples[:]
                    current_target_samples = []

                avg_gx = sum(x for x, y in samples_copy) / len(samples_copy)
                avg_gy = sum(y for x, y in samples_copy) / len(samples_copy)

                collected_pairs.append(((avg_gx, avg_gy), target_xy))
                print(f"Collected {target_name}: gaze=({avg_gx:.4f}, {avg_gy:.4f}) -> screen={target_xy}")

                with state_lock:
                    current_target_index += 1
                    last_target_switch_time = time.time()

                    if current_target_index >= len(calibration_targets):
                        A_x, A_y = fit_mapping(collected_pairs)
                        calibration_done = True
                        mapping_ready = True
                        experiment_start_time = time.time()
                        status_text = "Calibration complete. Move your eyes/head. Press Esc to quit."

                        calibration_finished_event.set()

                        print("\nCalibration complete.")
                        print("A_x =", A_x)
                        print("A_y =", A_y)

                continue

            with state_lock:
                local_mapping_ready = mapping_ready

            if local_mapping_ready:
                pt = apply_mapping(gaze_angle_x, gaze_angle_y)
                if pt is not None:
                    with state_lock:
                        current_point[0] = int((1 - SMOOTHING_ALPHA) * current_point[0] + SMOOTHING_ALPHA * pt[0])
                        current_point[1] = int((1 - SMOOTHING_ALPHA) * current_point[1] + SMOOTHING_ALPHA * pt[1])
                        x = current_point[0]
                        y = current_point[1]

                    update_aoi_state(x, y)
                    update_fixation_state(x, y)

                    current_label = detect_aoi(x, y)
                    with state_lock:
                        status_text = f"Looking at: {current_label} | Press Esc to quit"


def stop_process(process):
    if process is None:
        return

    try:
        process.terminate()
        process.wait(timeout=5)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def draw_loop():
    global SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX, current_point
    global current_aoi, aoi_entry_time

    root = tk.Tk()

    #Comment this for three monitor setup, otherwise it will take the full width of all three monitors
    #Uncomment for single monitor setup so it will take the full width of the monitor
    # SCREEN_WIDTH_PX = root.winfo_screenwidth()
    # SCREEN_HEIGHT_PX = root.winfo_screenheight()


    current_point = [SCREEN_WIDTH_PX // 2, SCREEN_HEIGHT_PX // 2]


    build_calibration_targets()
    build_aois()
    targets_ready.set()

    root.title("OpenFace AOI Demo")
    root.attributes("-fullscreen", True)
    root.configure(bg="black")
    root.focus_force()

    canvas = tk.Canvas(
        root,
        width=SCREEN_WIDTH_PX,
        height=SCREEN_HEIGHT_PX,
        bg="black",
        highlightthickness=0
    )
    canvas.pack(fill="both", expand=True)

    aoi_rects = {}
    for name, (x1, y1, x2, y2) in AOIS.items():
        rect = canvas.create_rectangle(x1, y1, x2, y2, outline="white", width=2)
        canvas.create_text((x1 + x2) // 2, 40, text=name, fill="white", font=("Arial", 24))
        aoi_rects[name] = rect

    cx = SCREEN_WIDTH_PX // 2
    cy = SCREEN_HEIGHT_PX // 2
    canvas.create_line(cx - 20, cy, cx + 20, cy, fill="gray", width=2)
    canvas.create_line(cx, cy - 20, cx, cy + 20, fill="gray", width=2)

    dot_radius = 12
    gaze_dot = canvas.create_oval(
        cx - dot_radius, cy - dot_radius,
        cx + dot_radius, cy + dot_radius,
        fill="red", outline="red",
        state="hidden"
    )

    calib_radius = 18
    calib_dot = canvas.create_oval(
        cx - calib_radius, cy - calib_radius,
        cx + calib_radius, cy + calib_radius,
        fill="yellow", outline="yellow"
    )

    label = canvas.create_text(
        20, 20,
        anchor="nw",
        fill="white",
        text="Starting...",
        font=("Arial", 18)
    )

    def update_canvas():
        with state_lock:
            local_current_aoi = current_aoi
            local_calibration_done = calibration_done
            local_target_index = current_target_index
            local_status_text = status_text
            local_x, local_y = current_point[0], current_point[1]

        for name, rect in aoi_rects.items():
            if name == local_current_aoi:
                canvas.itemconfig(rect, outline="lime", width=4)
            else:
                canvas.itemconfig(rect, outline="white", width=2)

        if not local_calibration_done and local_target_index < len(calibration_targets):
            _, (tx, ty) = calibration_targets[local_target_index]
            canvas.coords(
                calib_dot,
                tx - calib_radius, ty - calib_radius,
                tx + calib_radius, ty + calib_radius
            )
            canvas.itemconfig(calib_dot, state="normal")
            canvas.itemconfig(gaze_dot, state="hidden")
            canvas.itemconfig(label, text=local_status_text)
            canvas.coords(label, tx + 30, ty + 30)
        else:
            canvas.itemconfig(calib_dot, state="hidden")
            canvas.itemconfig(label, text="Calibration complete.")
            root.after(300, root.destroy)
            return

        root.after(16, update_canvas)

    def shutdown():
        global current_aoi, aoi_entry_time

        if not root.winfo_exists():
            return

        stop_event.set()

        with state_lock:
            if current_aoi and aoi_entry_time:
                log_visit(current_aoi, aoi_entry_time, time.time())
                current_aoi = None
                aoi_entry_time = None

        finalize_fixation()
        root.destroy()

    def on_escape(event=None):
        shutdown()

    root.bind("<Escape>", on_escape)
    root.protocol("WM_DELETE_WINDOW", shutdown)

    update_canvas()
    root.mainloop()

def finalize_aoi_visit():
    global current_aoi, aoi_entry_time

    with state_lock:
        if current_aoi is not None and aoi_entry_time is not None:
            log_visit(current_aoi, aoi_entry_time, time.time())
            current_aoi = None
            aoi_entry_time = None

def main():
    global experiment_start_time
    global current_target_index, current_target_samples, last_target_switch_time
    global calibration_done, mapping_ready, current_aoi, pending_aoi, pending_aoi_start, aoi_entry_time
    global fixation_active, fixation_start_time, fixation_center, fixation_points
    global A_x, A_y, latest_angles

    stop_event.clear()
    targets_ready.clear()
    experiment_start_time = None

    A_x = None
    A_y = None
    latest_angles = [None, None]

    current_target_index = 0
    current_target_samples = []
    last_target_switch_time = time.time()

    calibration_done = False
    mapping_ready = False

    current_aoi = None
    pending_aoi = None
    pending_aoi_start = None
    aoi_entry_time = None

    fixation_active = False
    fixation_start_time = None
    fixation_center = None
    fixation_points = []

    init_logs()

    process = None
    reader_thread = None

    try:
        process = start_openface()
        wait_for_csv()
        fieldnames = read_header()

        reader_thread = threading.Thread(
            target=read_csv_live,
            args=(fieldnames,),
            daemon=True
        )
        reader_thread.start()

        draw_loop()
        print("Calibration finished. Tracking in background... q to stop.")

        while not stop_event.is_set():
            if keyboard.is_pressed("q"):
                print("Stop key pressed. Ending experiment...")
                stop_event.set()
                break
            time.sleep(0.1)

    finally:
        stop_event.set()

        if reader_thread is not None and reader_thread.is_alive():
            reader_thread.join(timeout=1.0)

        finalize_aoi_visit()
        finalize_fixation()
        write_fixation_summary()

        if process is not None:
            stop_process(process)

if __name__ == "__main__":
    main()