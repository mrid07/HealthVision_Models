
from flask import Flask, jsonify, render_template, request, Response
from engine.engine import start_engine, stop_engine
import engine.state as state
import cv2
import time
import json
import numpy as np

app = Flask(__name__)

# --- ROUTES ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/math-stress")
def math_stress():
    return render_template("math_stress.html")

@app.route("/start", methods=["POST"])
def start():
    data = request.get_json(silent=True)
    user_name = "Guest"
    if data and "user_name" in data:
        user_name = data["user_name"]
        
    start_engine(user_name=user_name)
    return jsonify({"status": "started", "user": user_name})

@app.route("/stop", methods=["POST"])
def stop():
    stop_engine()
    return jsonify({"status": "stopped"})

@app.route("/data")
def data():
    # Return current analysis data
    # simplejson handles NaN -> null or we can convert manually if needed
    # standard json lib produces invalid json for nan/inf, but flask.jsonify might handle or fail.
    # We will sanitize NaNs to null just in case.
    
    with state.lock:
        # Create a copy
        data_copy = state.current_data.copy()
        
    # Sanitize
    for k, v in data_copy.items():
        if isinstance(v, float) and np.isnan(v):
            data_copy[k] = None
            
    return jsonify(data_copy)

def gen_frames():
    while True:
        frame = None
        with state.lock:
            if state.last_frame is not None:
                frame = state.last_frame.copy()
        
        if frame is None:
            # yield placeholder blank frame?
            # create blank
            frame = np.zeros((360, 480, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting for camera...", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        # Encode
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
        time.sleep(0.04) # cap at 25 fps for streaming

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, threaded=True, use_reloader=False) 
    # use_reloader=False to prevent double execution of threads if put in main scope, 
    # though here threads start on /start request.
