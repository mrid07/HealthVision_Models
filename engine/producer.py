
import cv2
import numpy as np
import time
import os
import engine.state as state
from engine.state import lock

def frame_producer(video_source=0):
    print(f"[Producer] Starting with source {video_source}")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("[Producer] Error opening video source.")
        return

    # Video Recorder Setup
    video_writer = None
    video_path = None
    if state.session_dir:
        video_path = os.path.join(state.session_dir, "video.mp4")
        # Defer initialization until we know frame size

    # Face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    face_box = None
    frame_count = 0
    FACE_DETECT_INTERVAL = 15
    
    while state.running:
        ret, frame = cap.read()
        if not ret:
            print("[Producer] Video ended or failed.")
            break
            
        t = time.time()
        frame_count += 1
        
        # Resize if large (optimization)
        h, w = frame.shape[:2]
        if max(h, w) > 1000:
            scale = 1000.0 / max(h, w)
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

        # Face detection
        if face_box is None or (frame_count % FACE_DETECT_INTERVAL == 0):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
            if len(faces) > 0:
                faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
                face_box = faces[0]
            else:
                face_box = None
                
        # ROI Extraction
        roi_mean_scalar = None
        disp = frame.copy()
        
        if face_box is not None:
            x, y, fw, fh = face_box
            
            # --- Define Multiple ROIs ---
            # 1. Forehead: top 5% down, 18% height, center 40% width
            r1 = (int(x + 0.3 * fw), int(y + 0.05 * fh), int(0.4 * fw), int(0.18 * fh))
            
            # 2. Left Cheek (approx): down 55%, left 15%, width 20%, height 15%
            r2 = (int(x + 0.15 * fw), int(y + 0.55 * fh), int(0.20 * fw), int(0.15 * fh))
            
            # 3. Right Cheek (approx): down 55%, right 65%, width 20%, height 15%
            r3 = (int(x + 0.65 * fw), int(y + 0.55 * fh), int(0.20 * fw), int(0.15 * fh))
            
            rois_to_process = [r1, r2, r3]
            pixel_accumulator = []

            for (rx, ry, rw, rh) in rois_to_process:
                # clamp coordinates
                rx = max(0, rx); ry = max(0, ry)
                if ry+rh > frame.shape[0]: rh = frame.shape[0] - ry
                if rx+rw > frame.shape[1]: rw = frame.shape[1] - rx

                if rw > 0 and rh > 0:
                    # Draw VISUAL rectangle
                    cv2.rectangle(disp, (rx, ry), (rx+rw, ry+rh), (0,255,0), 2)
                    
                    # Extract pixels
                    roi_crop = frame[ry:ry+rh, rx:rx+rw]
                    pixel_accumulator.append(np.reshape(roi_crop, (-1, 3)))

            # Compute combined mean if we have valid ROIs
            if pixel_accumulator:
                # Stack all pixels from all ROIs vertically
                all_pixels = np.vstack(pixel_accumulator)
                mean_rgb = np.mean(all_pixels, axis=0) # [B, G, R]
                mean_rgb = mean_rgb[::-1] # [R, G, B]
                roi_mean_scalar = mean_rgb[1] # Green channel
        
        # Update Shared State
        with lock:
            state.last_frame = disp
            
            if roi_mean_scalar is not None:
                 state.roi_signal.append(roi_mean_scalar)
                 state.time_buffer.append(t)
            else:
                # pad with last value or 0 to keep timing consistent
                if len(state.roi_signal) > 0:
                    val = state.roi_signal[-1]
                    state.roi_signal.append(val)
        
        # Write to video file
        if video_path and state.running:
            if video_writer is None:
                fh, fw = disp.shape[:2]
                try: 
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (fw, fh))
                    print(f"[Producer] Recording to {video_path}")
                except Exception as e:
                    print(f"[Producer] Video Writer Error: {e}")
                    video_writer = None 
            
            if video_writer is not None and video_writer.isOpened():
                video_writer.write(disp)
        
        time.sleep(0.005)

    # Cleanup
    if cap.isOpened():
        cap.release()
    if video_writer is not None and video_writer.isOpened():
        video_writer.release()
        print("[Producer] Video Saved.")
    
    print("[Producer] Stopped.")
