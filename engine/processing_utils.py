
import numpy as np
from scipy import signal, stats

# --- CONFIG ---
HPF = 0.7
LPF = 4.0

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
    # returns dominant_freq (Hz), power_at_peak, total_power, snr
    if len(sig) < 4:
        return np.nan, np.nan, np.nan, np.nan
    f, Pxx = signal.welch(sig, fs=fs, nperseg=min(256, len(sig)))
    # only consider HR band
    idx = np.where((f >= HPF) & (f <= LPF))[0]
    if len(idx) == 0:
        return np.nan, np.nan, np.nan, np.nan
    f_band = f[idx]
    P_band = Pxx[idx]
    peak_idx = np.argmax(P_band)
    peak_freq = f_band[peak_idx]
    peak_power = P_band[peak_idx]
    total_power = np.sum(Pxx)
    snr = peak_power / (np.sum(Pxx) - peak_power + 1e-8)
    # Return 4 values to match usage in compute_sqi
    return peak_freq, peak_power, total_power, snr

def compute_sqi(sig, fs):
    """
    Simple SQI: ratio of power in HR band to total power (0..1),
    higher means more plausible cardiac signal.
    """
    try:
        _, peak_power, total_power, snr = compute_psd_metrics(sig, fs)
    except ValueError:
         # handle case where compute_psd_metrics returns 3 values (if I messed up copy)
         # I checked above, it returns 4.
         return 0.0

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
