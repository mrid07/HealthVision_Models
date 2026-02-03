
import threading
import time
import os
import datetime
import engine.state as state
from engine.producer import frame_producer
from engine.processor import processing_loop

_producer_thread = None
_processor_thread = None

def start_engine(user_name="Guest"):
    global _producer_thread, _processor_thread
    
    if state.running:
        print("[Engine] Already running.")
        return

    print(f"[Engine] Starting for user: {user_name}...")
    state.running = True
    
    # Setup Session Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_folder_name = f"{user_name}_{timestamp}"
    state.session_dir = os.path.join("records", session_folder_name)
    
    os.makedirs(state.session_dir, exist_ok=True)
    state.user_name = user_name
    state.session_start_time = timestamp
    
    print(f"[Engine] Session directory created: {state.session_dir}")

    # Reset state? Maybe clear buffers?
    with state.lock:
        state.roi_signal.clear()
        state.time_buffer.clear()
        state.hr_history.clear()
        state.current_data = {
            "timestamp": None,
            "hr": float('nan'),
            "rr": float('nan'),
            "sys": float('nan'),
            "dia": float('nan'),
            "sqi": 0.0,
            "calmness_score": 0.0,
            "stress_label": "Unknown",
            "rmssd": 0.0,
            "sdnn": 0.0
        }

    _producer_thread = threading.Thread(target=frame_producer, args=(0,), daemon=True)
    _processor_thread = threading.Thread(target=processing_loop, daemon=True)
    
    _producer_thread.start()
    _processor_thread.start()
    print("[Engine] Threads started.")

def stop_engine():
    global _producer_thread, _processor_thread
    print("[Engine] Stopping...")
    state.running = False
    
    if _producer_thread:
        _producer_thread.join(timeout=2.0)
    if _processor_thread:
        _processor_thread.join(timeout=2.0)
        
    print("[Engine] Stopped.")
