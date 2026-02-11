
import time
import numpy as np
import traceback
import csv
import os
import engine.state as state
from engine.processing_utils import (
    compute_sqi,
    feature_vector_from_window,
    bandpass_filter,
    detrend_zscore,
    compute_hrv_metrics_from_hr_series,
    estimate_stress_and_calmness,
    pos_method
)
from engine.predictor import predict_metrics
from engine.bp_new_model import NewBPPredictor

# --- CONSTANTS ---
WINDOW_SIZE = 10.0       # Seconds of data required
PRED_INTERVAL = 1.0      # Update prediction every 1 sec
MIN_SQI = 0.10           # Minimum quality to update HR

bp_model_path = os.path.join(os.getcwd(), 'bp_model_ppg.pth')
bp_predictor = NewBPPredictor(bp_model_path)

def processing_loop():
    print("[Processor] Loop started.")
    
    while state.running:
        start_process_time = time.time()
        
        # 1. SNAPSHOT DATA (Fast, Lock held briefly)
        with state.lock:
            # Copy to list to avoid "deque mutated during iteration"
            ts_snapshot = list(state.time_buffer)
            sig_snapshot = list(state.roi_signal)
            green_sig_snapshot = list(state.green_signal)
            green_ts_snapshot = list(state.green_time_buffer)
        
        # 2. CHECK BUFFER SIZE
        has_face_data = len(ts_snapshot) >= 30
        has_finger_data = len(green_ts_snapshot) >= 30

        if not has_face_data and not has_finger_data:
            # Not enough samples yet
            time.sleep(0.5)
            continue
            
        # Determine strict window (last 10 seconds) for Face
        t_latest = time.time()
        if has_face_data: t_latest = ts_snapshot[-1]
        elif has_finger_data: t_latest = green_ts_snapshot[-1]

        # Init update variables
        hr_val = np.nan
        rr_val = np.nan
        sys_val = np.nan
        dia_val = np.nan
        sqi_val = 0.0
        stress_label = state.current_data["stress_label"] # keep last
        calmness_idx = state.current_data["calmness_score"]
        rmssd, sdnn = None, None
        sys_new, dia_new = state.current_data["sys_new"], state.current_data["dia_new"] # keep last

        # --- PHASE 2: Signal Processing (FACE) ---
        if has_face_data:
            ts_arr = np.array(ts_snapshot)
            sig_arr = np.array(sig_snapshot)
            
            # Windowing
            mask = (ts_arr >= (ts_snapshot[-1] - WINDOW_SIZE))
            window_ts = ts_arr[mask]
            window_sig = sig_arr[mask]
            
            duration = window_ts[-1] - window_ts[0] if len(window_ts) > 1 else 0
            
            if duration < (WINDOW_SIZE * 0.9):
                # Buffering
                if len(ts_snapshot) % 30 == 0:
                    print(f"[Processor] Buffering Face... {duration:.1f}s")
                stress_label = f"Buffering ({int(duration)}s)"
            else:
                try:
                    # 1. Preprocessing (POS Algorithm)
                    if window_sig.ndim == 2 and window_sig.shape[1] == 3:
                        raw_pulse = pos_method(window_sig)
                    else:
                        if window_sig.ndim == 2: raw_pulse = window_sig[:, 1]
                        else: raw_pulse = window_sig

                    processed = detrend_zscore(raw_pulse)
                    fs_est = len(window_sig) / duration if duration > 0 else 30.0
                    filtered = bandpass_filter(processed, fs_est, 0.7, 4.0)

                    # 2. SQI Check
                    fn_sqi = compute_sqi(filtered, fs_est)
                    sqi_val = fn_sqi
                    
                    if fn_sqi < MIN_SQI:
                        stress_label = "Low Signal"
                    else:
                        # Good Signal -> Run Models
                        feats = feature_vector_from_window(filtered, fs_est)
                        preds = predict_metrics(feats)
                        
                        hr_val = preds.get('hr', float('nan'))
                        rr_val = preds.get('rr', float('nan'))
                        sys_val = preds.get('sys', float('nan'))
                        dia_val = preds.get('dia', float('nan'))
                        
                        # Update History for HRV
                        if not np.isnan(hr_val):
                            with state.lock:
                                state.hr_history.append(hr_val)
                        
                        # Compute HRV
                        with state.lock:
                            if len(state.hr_history) > 3:
                                hrv_hist = list(state.hr_history)
                                rmssd, sdnn, stress_index = compute_hrv_metrics_from_hr_series(hrv_hist)

                        if stress_index is not None:
                            # Stress
                            s_res = estimate_stress_and_calmness(stress_index, rmssd)
                            stress_label, calmness_idx, _ = s_res 

                except Exception as e:
                    print(f"[Processor] Face Error: {e}")
                    traceback.print_exc()

        # --- PHASE 3: Signal Processing (FINGER - NEW BP) ---
        if has_finger_data:
            try:
                # Mimic user script: max 500 samples
                g_sig = green_sig_snapshot[-500:]
                g_ts = green_ts_snapshot[-500:]
                if len(g_ts) > 1:
                    duration_g = g_ts[-1] - g_ts[0]
                    if duration_g > 0:
                        fs_g = len(g_sig) / duration_g
                        # Predict
                        s_new, d_new = bp_predictor.predict(g_sig, fs_g)
                        if s_new is not None:
                            sys_new, dia_new = s_new, d_new
            except Exception as e:
                print(f"[Processor] Finger Error: {e}")

        # 3. WRITE TO STATE (Lock held briefly)
        with state.lock:
            state.current_data.update({
                "timestamp": t_latest,
                "hr": round(float(hr_val), 1) if not np.isnan(hr_val) else None,
                "rr": round(float(rr_val), 1) if not np.isnan(rr_val) else None,
                "sys": int(sys_val) if not np.isnan(sys_val) else None,
                "dia": int(dia_val) if not np.isnan(dia_val) else None,
                "sys_new": round(float(sys_new), 1) if sys_new is not None else None,
                "dia_new": round(float(dia_new), 1) if dia_new is not None else None,
                "sqi": round(float(sqi_val), 2),
                "calmness_score": round(float(calmness_idx), 2),
                "stress_label": stress_label,
                "rmssd": round(float(rmssd), 1) if rmssd else 0.0,
                "sdnn": round(float(sdnn), 1) if sdnn else 0.0
            })
        
        # Log to CSV
        if state.session_dir:
            csv_path = os.path.join(state.session_dir, "metrics.csv")
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Timestamp", "HR", "RR", "Sys", "Dia", "SysNew", "DiaNew", "SQI", "Stress", "Calmness"])
                
                writer.writerow([
                    t_latest,
                    state.current_data["hr"],
                    state.current_data["rr"],
                    state.current_data["sys"],
                    state.current_data["dia"],
                    state.current_data["sys_new"],
                    state.current_data["dia_new"],
                    state.current_data["sqi"],
                    state.current_data["stress_label"],
                    state.current_data["calmness_score"]
                ])

        print(f"[Processor] Updated: HR={hr_val:.1f}, BP_New={sys_new}/{dia_new}")

        # Sleep to maintain ~1 update per second
        elapsed = time.time() - start_process_time
        sleep_time = max(0.0, PRED_INTERVAL - elapsed)
        time.sleep(sleep_time)
