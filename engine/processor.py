
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
    estimate_stress_and_calmness
)
from engine.predictor import predict_metrics

# --- CONSTANTS ---
WINDOW_SIZE = 10.0       # Seconds of data required
PRED_INTERVAL = 1.0      # Update prediction every 1 sec
MIN_SQI = 0.10           # Minimum quality to update HR

def processing_loop():
    print("[Processor] Loop started.")
    
    while state.running:
        start_process_time = time.time()
        
        # 1. SNAPSHOT DATA (Fast, Lock held briefly)
        with state.lock:
            # Copy to list to avoid "deque mutated during iteration"
            ts_snapshot = list(state.time_buffer)
            sig_snapshot = list(state.roi_signal)
        
        # 2. CHECK BUFFER SIZE
        if len(ts_snapshot) < 30:
            # Not enough samples yet
            time.sleep(0.5)
            continue
            
        ts_arr = np.array(ts_snapshot)
        sig_arr = np.array(sig_snapshot)
        
        # Determine strict window (last 10 seconds)
        t_latest = ts_arr[-1]
        mask = (ts_arr >= (t_latest - WINDOW_SIZE))
        
        window_ts = ts_arr[mask]
        window_sig = sig_arr[mask]
        
        # Calculate approximate FPS in this window
        duration = window_ts[-1] - window_ts[0] if len(window_ts) > 1 else 0
        
        # --- PHASE 1: Wait for Buffer Fill (10s) ---
        if duration < (WINDOW_SIZE * 0.9):
            # We are still filling the buffer (e.g. at 5s out of 10s)
            # Update SQI only, but set HR to NaN to indicate "Loading"
            if len(ts_snapshot) % 30 == 0: # Print occasionally
                print(f"[Processor] Buffering... {duration:.1f}s / {WINDOW_SIZE}s")
            
            with state.lock:
                state.current_data["sqi"] = 0.0
                state.current_data["stress_label"] = f"Buffering ({int(duration)}s)"
            
            time.sleep(PRED_INTERVAL)
            continue

        # --- PHASE 2: Signal Processing ---
        try:
            # 1. Preprocessing
            processed = detrend_zscore(window_sig)
            
            # (Optional) Bandpass for HR range
            fs_est = len(window_sig) / duration if duration > 0 else 30.0
            filtered = bandpass_filter(processed, fs_est, 0.7, 4.0)

            # 2. SQI Check
            fn_sqi = compute_sqi(filtered, fs_est)
            
            # --- PHASE 3: Prediction or Rejection ---
            if fn_sqi < MIN_SQI:
                msg = f"Low Signal Quality ({fn_sqi:.2f} < {MIN_SQI})"
                # print(f"[Processor] {msg}")
                # We update SQI so user knows why it's stuck, but keep old HR/BP
                with state.lock:
                    state.current_data["sqi"] = float(fn_sqi)
                    state.current_data["stress_label"] = "Low Signal"
            else:
                # Good Signal -> Run Models
                feats = feature_vector_from_window(filtered, fs_est)
                
                # Predict
                preds = predict_metrics(feats) # Returns {hr, rr, sys, dia}
                
                # Update History for HRV
                hr_val = preds.get('hr', float('nan'))
                
                if not np.isnan(hr_val):
                    with state.lock:
                         state.hr_history.append(hr_val)
                
                # Compute HRV
                rmssd, sdnn, stress_index = None, None, None
                with state.lock:
                     # Calculate HRV if we have enough history
                     if len(state.hr_history) > 3:
                         hrv_hist = list(state.hr_history)
                         rmssd, sdnn, stress_index = compute_hrv_metrics_from_hr_series(hrv_hist)

                # Stress
                stress_res = estimate_stress_and_calmness(stress_index, rmssd)
                stress_label, calmness_idx, calmness_score = stress_res 
                # estimate_stress_and_calmness returns (label, calmness_index, calmness_score)

                # 3. WRITE TO STATE (Lock held briefly)
                with state.lock:
                    state.current_data.update({
                        "timestamp": t_latest,
                        "hr": round(float(hr_val), 1) if not np.isnan(hr_val) else None,
                        "rr": round(float(preds.get('rr', float('nan'))), 1),
                        "sys": int(preds.get('sys', 0)) if not np.isnan(preds.get('sys')) else None,
                        "dia": int(preds.get('dia', 0)) if not np.isnan(preds.get('dia')) else None,
                        "sqi": round(float(fn_sqi), 2),
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
                            writer.writerow(["Timestamp", "HR", "RR", "Sys", "Dia", "SQI", "Stress", "Calmness"])
                        
                        writer.writerow([
                            t_latest,
                            state.current_data["hr"],
                            state.current_data["rr"],
                            state.current_data["sys"],
                            state.current_data["dia"],
                            state.current_data["sqi"],
                            state.current_data["stress_label"],
                            state.current_data["calmness_score"]
                        ])

                print(f"[Processor] Updated: HR={hr_val:.1f}, SQI={fn_sqi:.2f}, FPS={fs_est:.1f}")

        except Exception as e:
            print(f"[Processor] Error: {e}")
            traceback.print_exc()

        # Sleep to maintain ~1 update per second
        elapsed = time.time() - start_process_time
        sleep_time = max(0.0, PRED_INTERVAL - elapsed)
        time.sleep(sleep_time)
