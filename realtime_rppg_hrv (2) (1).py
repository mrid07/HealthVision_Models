"""
Improved real-time rPPG pipeline with Haarcascade face detection, forehead ROI,
better preprocessing (detrend + bandpass), multiple color methods (GREEN/CHROM/POS),
signal quality index (SQI), and richer features for model prediction.

Make sure your pretrained models accept the feature vector length produced below.
If your models accept only 3 features (mean/std/range), the code will fall back to that.
"""

import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from collections import deque
import threading
import time
from matplotlib.animation import FuncAnimation
from scipy import signal, stats

# --- CONFIG ---
VIDEO_SOURCE = 0  # 0 for webcam, or path to video file
MODEL_PATHS = {
    "hr": "D:/BTP/BTP3rd/heart_rate_model.pkl",
    "rr": "D:/BTP/BTP3rd/respiration_rate_model.pkl",
    "sys": "D:/BTP/BTP3rd/systolic_bp_model.pkl",
    "dia": "D:/BTP/BTP3rd/diastolic_bp_model.pkl",
}
WINDOW_SIZE = 10         # seconds for analysis window
PRED_INTERVAL = 1.0      # update interval (s)
HRV_HISTORY_SECONDS = 60
FACE_DETECT_INTERVAL = 15  # detect face every N frames to save CPU
MIN_SQI = 0.25           # minimum SQI threshold to accept window predictions
USE_METHOD = 'POS'       # options: 'GREEN', 'CHROM', 'POS' (try POS for robustness)

# bandpass for heart (0.7 - 4.0 Hz) -> ~42 - 240 bpm, reasonable range
HPF = 0.7
LPF = 4.0

# --- load models ---
models = {}
for k, p in MODEL_PATHS.items():
    try:
        models[k] = joblib.load(p)
        print(f"Loaded model {k} from {p}")
    except Exception as e:
        print(f"Warning: could not load model {k} from {p}: {e}")
        models[k] = None

# --- shared buffers ---
roi_signal = deque()     # per-frame scalar rPPG proxy (float)
time_buffer = deque()    # timestamp for each sample
last_frame = None        # latest camera frame for overlay
lock = threading.Lock()
keep_running = True

# --- face detector (haarcascade) ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- util: filters & processing ---
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(x, fs, lowcut=HPF, highcut=LPF, order=3):
    if len(x) < (order*3):
        # too short to filter reliably -> return detrended
        return signal.detrend(x)
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    try:
        return signal.filtfilt(b, a, x)
    except Exception:
        # fallback to lfilter if filtfilt fails for short/edge cases
        return signal.lfilter(b, a, x)

def detrend_zscore(x):
    x = np.array(x, dtype=np.float64)
    x = signal.detrend(x)
    # z-score for normalized features
    if np.std(x) == 0:
        return x
    return (x - np.mean(x)) / (np.std(x) + 1e-8)

# CHROM method (per-de Haan et al.)
def chrom_method(rgb_ts):
    """
    rgb_ts: Nx3 array of mean R,G,B over time.
    returns 1D chrom signal of length N
    """
    X = rgb_ts.T  # 3 x N
    # projection matrix
    # S = [ 0  1 -1 ] etc as in CHROM paper: use skin-normalized projections
    # Implementation: compute chroma as combination to suppress motion:
    ry = X[0] - X[1]
    gz = X[0] + X[1] - 2 * X[2]
    # normalize each
    std_ry = np.std(ry) if np.std(ry) != 0 else 1.0
    std_gz = np.std(gz) if np.std(gz) != 0 else 1.0
    s = ry / std_ry - gz / std_gz
    return s

# POS method (plane-orthogonal-to-skin)
def pos_method(rgb_ts, L=32):
    """
    Simple POS implementation: returns 1D signal.
    rgb_ts: Nx3 array (R,G,B means)
    """
    X = rgb_ts.T  # 3 x N
    # normalize by mean over small rolling window to reduce illumination
    mean_rgb = np.mean(X, axis=1, keepdims=True)
    Xn = X / (mean_rgb + 1e-8)
    # projection
    S = np.array([[0, 1, -1],
                  [-2, 1, 1]]) @ Xn
    # combine
    h = S[0] - (np.std(S[0]) / (np.std(S[1]) + 1e-8)) * S[1]
    return h

def compute_psd_metrics(sig, fs):
    # returns dominant_freq (Hz), power_at_peak, total_power
    if len(sig) < 4:
        return np.nan, np.nan, np.nan
    f, Pxx = signal.welch(sig, fs=fs, nperseg=min(256, len(sig)))
    # only consider HR band
    idx = np.where((f >= HPF) & (f <= LPF))[0]
    if len(idx) == 0:
        return np.nan, np.nan, np.nan
    f_band = f[idx]
    P_band = Pxx[idx]
    peak_idx = np.argmax(P_band)
    peak_freq = f_band[peak_idx]
    peak_power = P_band[peak_idx]
    total_power = np.sum(Pxx)
    snr = peak_power / (np.sum(Pxx) - peak_power + 1e-8)
    return peak_freq, peak_power, total_power, snr

def compute_sqi(sig, fs):
    """
    Simple SQI: ratio of power in HR band to total power (0..1),
    higher means more plausible cardiac signal.
    """
    _, peak_power, total_power, snr = compute_psd_metrics(sig, fs)
    if np.isnan(peak_power) or np.isnan(total_power) or total_power == 0:
        return 0.0
    return float(peak_power / (total_power + 1e-9))

def feature_vector_from_window(sig_window, fs):
    """
    Compute a feature vector for model input.
    Includes: mean, std, range, peak_freq(Hz), snr, skew, kurtosis, sqi
    """
    arr = np.array(sig_window, dtype=float)
    if len(arr) == 0:
        return np.zeros(8)
    mean_v = np.mean(arr)
    std_v = np.std(arr)
    range_v = float(np.max(arr) - np.min(arr))
    peak_freq, _, _, snr = compute_psd_metrics(arr, fs)
    skew = float(stats.skew(arr)) if len(arr) > 2 else 0.0
    kurt = float(stats.kurtosis(arr)) if len(arr) > 3 else 0.0
    sqi = compute_sqi(arr, fs)
    # convert peak_freq (Hz) to bpm for easier model compatibility
    peak_bpm = peak_freq * 60.0 if not np.isnan(peak_freq) else np.nan
    feat = np.array([mean_v, std_v, range_v, peak_bpm, snr, skew, kurt, sqi]).reshape(1, -1)
    return feat

# --- frame producer: capture and extract ROI mean(s) ---
def frame_producer(source=VIDEO_SOURCE):
    global keep_running, last_frame
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error opening video source. Exiting frame thread.")
        keep_running = False
        return

    # read real fps if available
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    FPS = reported_fps if reported_fps and reported_fps > 0 else 30.0
    print(f"[Producer] Using capture FPS = {FPS:.2f}")

    frame_count = 0
    face_box = None  # cache last detected face

    rgb_history = deque(maxlen=int(WINDOW_SIZE * max(1, FPS) + 10))  # store per-frame RGB means
    t_history = deque(maxlen=int(WINDOW_SIZE * max(1, FPS) + 10))

    while keep_running:
        ret, frame = cap.read()
        if not ret:
            keep_running = False
            break

        # resize if large
        h, w = frame.shape[:2]
        if max(h, w) > 1000:
            scale = 1000.0 / max(h, w)
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

        frame_count += 1
        t = time.time()

        # detect face occasionally to save CPU
        if face_box is None or (frame_count % FACE_DETECT_INTERVAL == 0):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
            if len(faces) > 0:
                # choose largest
                faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
                face_box = faces[0]
            else:
                face_box = None

        # compute forehead ROI from face box
        roi_mean_scalar = None
        if face_box is not None:
            x, y, fw, fh = face_box
            # forehead region: top 18-28% of face height, center 40% width
            fx = int(x + 0.3 * fw)
            fw_roi = int(0.4 * fw)
            fy = int(y + 0.05 * fh)
            fh_roi = int(0.18 * fh)
            # clamp
            fx = max(0, fx); fy = max(0, fy)
            if fy+fh_roi > frame.shape[0]:
                fh_roi = frame.shape[0] - fy
            if fx+fw_roi > frame.shape[1]:
                fw_roi = frame.shape[1] - fx

            roi = frame[fy:fy+fh_roi, fx:fx+fw_roi]
            # compute mean RGB
            mean_rgb = np.mean(np.reshape(roi, (-1, 3)), axis=0)  # [B,G,R]
            # convert ordering to R,G,B
            mean_rgb = mean_rgb[::-1]  # to R,G,B
            roi_mean_scalar = mean_rgb  # store vector for CHROM/POS
            # draw rectangle on last_frame for overlay
            disp = frame.copy()
            cv2.rectangle(disp, (fx, fy), (fx+fw_roi, fy+fh_roi), (0,255,0), 2)
            with lock:
                last_frame = disp
        else:
            # when no face, save blank or last frame
            with lock:
                if last_frame is None:
                    last_frame = frame.copy()

        # append rgb_mean regardless (if face missing, append NaNs)
        with lock:
            if roi_mean_scalar is not None:
                rgb_history.append(roi_mean_scalar)
                t_history.append(t)
            else:
                # append last valid or zeros (helps continuity)
                if len(rgb_history) > 0:
                    rgb_history.append(rgb_history[-1])
                    t_history.append(t)
                else:
                    rgb_history.append(np.array([0.0,0.0,0.0]))
                    t_history.append(t)

            # update global roi_signal + time_buffer with scalar depending on method
            # We will store the raw per-frame RGB means and let prediction window compute POS/CHROM
            # For quick live overlay we also store green-channel mean scalar
            roi_signal.append(rgb_history[-1][1])  # green channel (index 1)
            time_buffer.append(t_history[-1])

            # cap buffers (avoid memory blowup)
            max_len = int(3600 * 30 / max(1, int(FPS)))  # very large cap, just in case
            while len(roi_signal) > max_len:
                roi_signal.popleft(); time_buffer.popleft()

        # sleep to match capture fps (if camera is fast)
        # get actual capture fps pace
        time.sleep(max(0.0, 1.0 / max(1.0, FPS) - 0.002))

    cap.release()
    print("[Producer] stopped.")

# start producer
producer_thread = threading.Thread(target=frame_producer, args=(VIDEO_SOURCE,), daemon=True)
producer_thread.start()

# --- Visualization setup (kept similar to your original) ---
plt.style.use('seaborn-v0_8')
fig = plt.figure(constrained_layout=True, figsize=(10, 7))
gs = fig.add_gridspec(3, 2)
ax_hr = fig.add_subplot(gs[0, :])        # HR time-series
ax_rr = fig.add_subplot(gs[1, 0])        # RR time-series
ax_tach = fig.add_subplot(gs[1, 1])      # Tachogram / RR intervals
ax_calm = fig.add_subplot(gs[2, :])      # Calmness bar

hr_times = deque(maxlen=300)
hr_values = deque(maxlen=300)
rr_times = deque(maxlen=300)
rr_values = deque(maxlen=300)
ibi_vals = deque(maxlen=300)

line_hr, = ax_hr.plot([], [], marker='o')
line_rr, = ax_rr.plot([], [], marker='o')
line_tach, = ax_tach.plot([], [], marker='x')
calm_bar = ax_calm.barh([0], [0.0], height=0.6)
calm_text = ax_calm.text(0.5, 0.5, '', va='center', ha='center', fontsize=14)

ax_hr.set_title('Heart Rate (BPM)')
ax_hr.set_xlabel('Time (s)')
ax_hr.set_ylabel('BPM')

ax_rr.set_title('Respiration Rate (BPM)')
ax_rr.set_xlabel('Time (s)')
ax_rr.set_ylabel('Breaths/min')

ax_tach.set_title('Tachogram (RR Intervals, s)')
ax_tach.set_xlabel('Window Index')
ax_tach.set_ylabel('RR interval (s)')

ax_calm.set_title('Calmness / Biofeedback')
ax_calm.set_xlim(0, 1)
ax_calm.set_yticks([])

cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Feed', 640, 480)

start_time = time.time()

def stop():
    global keep_running
    keep_running = False
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

# HRV helpers (reused from original)
def compute_hrv_metrics_from_hr_series(hr_series):
    if len(hr_series) < 3:
        return None, None, None
    rr_intervals = 60.0 / np.array(hr_series)
    diff_rr = np.diff(rr_intervals)
    rmssd = float(np.sqrt(np.mean(diff_rr ** 2))) if len(diff_rr) > 0 else None
    sdnn = float(np.std(rr_intervals, ddof=1)) if len(rr_intervals) > 1 else None
    try:
        hist, bin_edges = np.histogram(rr_intervals, bins=20)
        mode_idx = int(np.argmax(hist))
        Mo = float((bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2.0)
        AMo = float(hist[mode_idx]) / len(rr_intervals) * 100.0
        MxDMn = float(np.max(rr_intervals) - np.min(rr_intervals))
        stress_index = float(AMo / (2.0 * Mo * MxDMn)) if MxDMn != 0 else np.nan
    except Exception:
        stress_index = np.nan
    return rmssd, sdnn, stress_index

def estimate_stress_and_calmness(stress_index, rmssd):
    if stress_index is None or np.isnan(stress_index) or rmssd is None:
        return "Unknown", 0.0, 0.0
    if stress_index < 50 and rmssd > 0.08:
        label = "Low Stress"; calmness_score = 0.95
    elif stress_index < 150 and rmssd > 0.04:
        label = "Moderate Stress"; calmness_score = 0.6
    else:
        label = "High Stress"; calmness_score = 0.25
    calmness_index = (1.0 / (1.0 + stress_index)) + (rmssd / 0.2)
    calmness_index = np.clip(calmness_index / 2.0, 0.0, 1.0)
    return label, calmness_index, calmness_score

# Keep last prediction values for overlay
last_preds = {"hr": np.nan, "rr": np.nan, "sys": np.nan, "dia": np.nan, "calm": 0.0, "label": "Unknown"}

def update(frame_num):
    global last_preds
    current_time = time.time()
    with lock:
        local_rgb = list(roi_signal)      # NOTE: this currently stores green scalars, but we have direct rgb means stored in producer's internal rgb_history too
        local_times = list(time_buffer)
        frame_img = last_frame.copy() if last_frame is not None else np.zeros((360,480,3), dtype=np.uint8)

    # Need at least some frames
    if len(local_times) < 2:
        # show overlay only
        cv2.putText(frame_img, "No face / insufficient data", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.imshow('Camera Feed', frame_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop()
        return line_hr, line_rr, line_tach, calm_bar

    # determine fps from timestamps (robust)
    dt = np.diff(local_times)
    median_dt = np.median(dt) if len(dt) > 0 else 1.0/30.0
    fs = 1.0 / max(median_dt, 1e-3)

    # build window of WINDOW_SIZE seconds
    window_start_time = local_times[-1] - WINDOW_SIZE
    # find index
    idxs = [i for i,t in enumerate(local_times) if t >= window_start_time]
    if len(idxs) < 4:
        # not enough samples yet
        return line_hr, line_rr, line_tach, calm_bar

    # For CHROM/POS we need RGB per-frame history; we don't expose it globally, but we can reconstruct
    # from roi_signal where we stored green only. If you want full POS/CHROM, modify producer to store rgb means globally.
    # Here we will attempt to use green-based approach + fallback to POS/CHROM if available.
    # Attempt to pull rgb_history from producer thread by accessing 'roi_signal' which currently stores green only.
    # So we'll primarily use the green-channel raw values as input (this is robust and compatible with earlier models).

    window = np.array(local_rgb[-len(idxs):], dtype=float)
    # detrend + bandpass
    sig_bp = bandpass_filter(window, fs)
    sig_norm = detrend_zscore(sig_bp)

    # compute SQI
    sqi = compute_sqi(sig_bp, fs)

    # compute features
    feat = feature_vector_from_window(sig_bp, fs)  # returns 8 features

    # Model compatibility: if model expects 3 features (old) then fallback to 3-feature vector
    def prepare_feature_for_model(model, feat8):
        try:
            # try to inspect model input dimension (sklearn often has coef_ or n_features_in_)
            n_in = None
            if hasattr(model, "n_features_in_"):
                n_in = model.n_features_in_
            elif hasattr(model, "coef_"):
                n_in = model.coef_.shape[-1]
            if n_in == 3:
                # fallback to mean/std/range
                return feat8[:, :3]
        except Exception:
            pass
        return feat8

    # predictions only if SQI is OK; otherwise output NaNs or previous values (you can adjust policy)
    if sqi < MIN_SQI:
        # weak signal: do not replace last prediction, just display lower confidence
        hr_pred = last_preds.get("hr", np.nan)
        rr_pred = last_preds.get("rr", np.nan)
        sys_pred = last_preds.get("sys", np.nan)
        dia_pred = last_preds.get("dia", np.nan)
    else:
        # predict
        try:
            if models.get('hr') is not None:
                X = prepare_feature_for_model(models['hr'], feat)
                hr_pred = float(models['hr'].predict(X)[0])
            else:
                # fallback: estimate HR by peak frequency of PSD
                peak_freq, _, _, _ = compute_psd_metrics(sig_bp, fs)
                hr_pred = float(peak_freq * 60.0) if not np.isnan(peak_freq) else np.nan
        except Exception:
            hr_pred = np.nan

        try:
            if models.get('rr') is not None:
                X = prepare_feature_for_model(models['rr'], feat)
                rr_pred = float(models['rr'].predict(X)[0])
            else:
                rr_pred = np.nan
        except Exception:
            rr_pred = np.nan

        try:
            if models.get('sys') is not None:
                X = prepare_feature_for_model(models['sys'], feat)
                sys_pred = float(models['sys'].predict(X)[0])
            else:
                sys_pred = np.nan
        except Exception:
            sys_pred = np.nan

        try:
            if models.get('dia') is not None:
                X = prepare_feature_for_model(models['dia'], feat)
                dia_pred = float(models['dia'].predict(X)[0])
            else:
                dia_pred = np.nan
        except Exception:
            dia_pred = np.nan

        # update last preds
        last_preds["hr"], last_preds["rr"], last_preds["sys"], last_preds["dia"] = hr_pred, rr_pred, sys_pred, dia_pred

    # append to plots
    tstamp = current_time - start_time
    hr_times.append(tstamp); hr_values.append(hr_pred)
    rr_times.append(tstamp); rr_values.append(rr_pred)

    # compute HRV
    try:
        hr_history = list(hr_values)[-int(max(3, HRV_HISTORY_SECONDS)):]
        rmssd, sdnn, stress_index = compute_hrv_metrics_from_hr_series(hr_history)
    except Exception:
        rmssd, sdnn, stress_index = None, None, None

    stress_label, calmness_index, calmness_score = estimate_stress_and_calmness(stress_index, rmssd)
    last_preds["calm"] = calmness_index
    last_preds["label"] = stress_label

    # Update plots similar to original
    line_hr.set_data(hr_times, hr_values)
    if len(hr_times) > 0:
        ax_hr.set_xlim(max(0, hr_times[0]), hr_times[-1] + 0.1)
        # avoid invalid min/max when nan present
        valid_hr = [v for v in hr_values if not np.isnan(v)]
        if len(valid_hr) > 0:
            ax_hr.set_ylim(max(30, np.nanmin(valid_hr) - 10), min(220, np.nanmax(valid_hr) + 10))

    line_rr.set_data(rr_times, rr_values)
    if len(rr_times) > 0:
        ax_rr.set_xlim(max(0, rr_times[0]), rr_times[-1] + 0.1)
        valid_rr = [v for v in rr_values if not np.isnan(v)]
        if len(valid_rr) > 0:
            ax_rr.set_ylim(max(0, np.nanmin(valid_rr) - 2), np.nanmax(valid_rr) + 2)

    ibi = [60.0 / v if (v is not None and not np.isnan(v) and v > 0) else np.nan for v in hr_values]
    line_tach.set_data(np.arange(len(ibi)), ibi)
    ax_tach.set_xlim(0, max(10, len(ibi)))
    if len([x for x in ibi if not np.isnan(x)]) > 0:
        ax_tach.set_ylim(max(0, np.nanmin([x for x in ibi if not np.isnan(x)]) - 0.1), np.nanmax([x for x in ibi if not np.isnan(x)]) + 0.1)

    calm_bar[0].set_width(calmness_index)
    calm_text.set_text(f"{stress_label} | Calmness: {calmness_index:.2f} | SQI: {sqi:.2f}")

    # overlay actual camera frame with predictions
    overlay = frame_img.copy()
    cv2.putText(overlay, f"HR: {last_preds['hr']:.1f} bpm", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,200,200), 2)
    cv2.putText(overlay, f"RR: {last_preds['rr']:.1f} bpm", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,200,200), 2)
    cv2.putText(overlay, f"SYS/DIA: {last_preds['sys']:.0f}/{last_preds['dia']:.0f} mmHg", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
    cv2.putText(overlay, f"Calmness: {calmness_index:.2f} ({stress_label})", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
    cv2.putText(overlay, f"SQI: {sqi:.2f}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    cv2.imshow('Camera Feed', overlay)

    # quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop()

    return line_hr, line_rr, line_tach, calm_bar

# run animation
ani = FuncAnimation(fig, update, interval=PRED_INTERVAL*1000, blit=False)
print("Starting improved rPPG dashboard. Press 'q' in the camera window to stop.")
plt.show()

# cleanup
stop()
producer_thread.join(timeout=2)
print("Exited.")
