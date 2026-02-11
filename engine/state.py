# engine/state.py

import threading
from collections import deque
import numpy as np

lock = threading.Lock()
running = False

# --- CONFIG ---
# These match the original script buffers
# 30 fps * 60 seconds = 1800 samples is enough for processing, 
# but original script used dynamic check. We'll set a safe maxlen.
MAX_BUFFER_LEN = 3000 

# --- SHARED BUFFERS ---
# Producer writes, Processor reads
roi_signal = deque(maxlen=MAX_BUFFER_LEN)      # Stores per-frame [R, G, B] means
green_signal = deque(maxlen=MAX_BUFFER_LEN)    # Stores per-frame Green mean (whole frame)
time_buffer = deque(maxlen=MAX_BUFFER_LEN)     # Stores timestamps for roi_signal
green_time_buffer = deque(maxlen=MAX_BUFFER_LEN) # Stores timestamps for green_signal
fps_buffer = deque(maxlen=100) # Optional, for producer to smooth fps

# Last frame for streaming (Producer writes, App reads)
last_frame = None

# --- PIPELINE STATE ---
# Processor updates this, App reads it
current_data = {
    "timestamp": None,
    "hr": np.nan,
    "rr": np.nan,
    "sys": np.nan,
    "dia": np.nan,
    "sys_new": np.nan, 
    "dia_new": np.nan,
    "sqi": 0.0,
    "calmness_score": 0.0,
    "stress_label": "Unknown",
    "rmssd": 0.0,
    "sdnn": 0.0
}

# History for HRV calculation (Processor usage)
hr_history = deque(maxlen=300) # Store recent HR values for HRV calc

# --- SESSION STATE ---
user_name = None
session_start_time = None
session_dir = None
