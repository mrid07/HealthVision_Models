
import torch
import torch.nn as nn
import numpy as np
from scipy.signal import butter, filtfilt, resample
import os

# ======================
# MODEL DEFINITION
# ======================

WINDOW_SEC = 8
MODEL_FS = 100
STEP_SEC = 2
WINDOW_SAMPLES = WINDOW_SEC * MODEL_FS

class BPModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.regressor = nn.Sequential(
            nn.Linear(128 * (WINDOW_SAMPLES // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)

# ======================
# HELPER CLASS
# ======================

class NewBPPredictor:
    def __init__(self, model_path):
        self.model = BPModel()
        try:
            # Map location cpu just in case
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.model.eval()
            self.loaded = True
            print(f"[NewBPPredictor] Model loaded from {model_path}")
        except Exception as e:
            print(f"[NewBPPredictor] Failed to load model: {e}")
            self.loaded = False

    def predict(self, signal_buffer, fs_input):
        """
        signal_buffer: list or array of green channel values (last ~8+ seconds)
        fs_input: estimated sampling rate of the buffer
        Returns: (sbp, dbp) or (None, None)
        """
        if not self.loaded or len(signal_buffer) < 2:
            return None, None

        signal = np.array(signal_buffer)
        
        # We need roughly WINDOW_SEC seconds. 
        # If we have more, take the last WINDOW_SEC seconds based on fs
        # But the original script takes the whole buffer duration. 
        # "if duration >= WINDOW_SEC" -> process "signal = np.array(buffer)"
        
        # Safety check on length
        if len(signal) < 30: 
            return None, None

        try:
            # Filter
            filtered = self.bandpass(signal, fs_input)

            # Resample to 100 Hz
            # We need exact WINDOW_SAMPLES points representing the last WINDOW_SEC
            # But the original code takes 'filtered' (which is the whole buffer)
            # and resamples it to WINDOW_SAMPLES.
            # This implies the original code assumes 'buffer' is EXACTLY extracting WINDOW_SEC worth of data?
            # Wait, in the original script:
            # duration = timestamps[-1] - timestamps[0]
            # if duration >= WINDOW_SEC:
            #     signal = np.array(buffer)
            #     fs_cam = len(signal) / duration
            #     filtered = bandpass(signal, fs_cam)
            #     resampled = resample(filtered, WINDOW_SAMPLES)
            
            # This logic SQUEEZES the entire buffer duration (which might be slightly > 8s) into 800 samples.
            # It treats the whole buffer as the window.
            # So we should pass a buffer that represents ~8 seconds.

            resampled = resample(filtered, WINDOW_SAMPLES)

            # Normalize
            resampled = (resampled - np.mean(resampled)) / np.std(resampled)

            # First derivative
            d1 = np.gradient(resampled)

            # Stack channels
            input_signal = np.vstack([resampled, d1])
            input_signal = torch.tensor(input_signal, dtype=torch.float32).unsqueeze(0)

            # Predict
            with torch.no_grad():
                pred = self.model(input_signal).numpy()[0]

            sbp, dbp = pred
            return float(sbp), float(dbp)

        except Exception as e:
            print(f"[NewBPPredictor] Prediction error: {e}")
            return None, None

    def bandpass(self, signal, fs):
        # 0.7 - 3 Hz
        nyq = 0.5 * fs
        low = 0.7 / nyq
        high = 3.0 / nyq
        # Safety check for filter bounds
        if low <= 0 or high >= 1:
             return signal # cannot filter
        b, a = butter(2, [low, high], btype='band')
        return filtfilt(b, a, signal)

