
import threading
import time
import engine.state as state
from engine.producer import frame_producer
from engine.processor import processing_loop

_producer_thread = None
_processor_thread = None

def start_engine():
    global _producer_thread, _processor_thread
    
    if state.running:
        print("[Engine] Already running.")
        return

    print("[Engine] Starting...")
    state.running = True
    
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
