from datetime import datetime
from flask import Flask, request, jsonify, render_template, Response, redirect, url_for, session, send_from_directory, flash
import cv2
import pickle
import os
import functools
import numpy as np
import pytz
from werkzeug.utils import secure_filename

app = Flask(__name__)  
app.secret_key = "supersecretkey"  # For session management

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Login Required Decorator
def login_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            flash("Please log in to access this page.")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/home")
@login_required
def Home():
    return render_template("home.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == "test" and password == "test":
            session["user"] = username
            flash("Login successful!", "success")  # Flash success message
            return '''
                <script>
                    alert('Login successful.');
                    window.location.href = '/home';
                </script>
                '''
        else:
            return '''
                <script>
                    alert('Login Failed ! Try again');
                    window.location.href = '/login';
                </script>
                '''
    return render_template("login.html")


import shutil  # Add this import

@app.route("/logout")
def logout():
    session.pop("user", None)
    # Remove the uploads folder
    if os.path.exists(app.config["UPLOAD_FOLDER"]):
        shutil.rmtree(app.config["UPLOAD_FOLDER"])
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)  # Recreate an empty uploads folder
    flash("Logged out successfully!")
    return redirect(url_for("login"))


@app.route("/upload", methods=["POST"])
@login_required
def upload_video():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    if file:
        filename = secure_filename(file.filename)
        local_file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(local_file_path)

        # Save the uploaded file locally
        session["filename"] = filename
        
        # Extract the first frame for ROI selection
        cap = cv2.VideoCapture(local_file_path)
        success, frame = cap.read()
        cap.release()
        if success:
            frame_path = os.path.join(app.config["UPLOAD_FOLDER"], "frame.jpg")
            cv2.imwrite(frame_path, frame)

        return redirect(url_for("select_parking"))

@app.route("/select_parking", methods=["GET", "POST"])
@login_required
def select_parking():
    if request.method == "GET":
        frame_url = url_for("uploaded_file", filename="frame.jpg")
        return render_template("select_parking.html", frame_url=frame_url)

    if request.method == "POST":
        roi_data = request.json.get("roi_data", [])
        roi_data = [
            { "x": int(r["x"]), "y": int(r["y"]), "width": int(r["width"]), "height": int(r["height"]) }
            for r in roi_data
        ]
        with open("CarParkingSpots", "wb") as f:
            pickle.dump(roi_data, f)
        return jsonify({"message": "Parking spots saved successfully!"})
@app.route("/uploads/<filename>")
@login_required
def uploaded_file(filename):
    """Serve files from the uploads directory."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

def generate_frames(filename):
    local_file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    # Load parking spot data
    if not os.path.exists("CarParkingSpots"):
        yield (b"ROI file not found. Please define parking spots.")
        return

    with open("CarParkingSpots", "rb") as f:
        posList = pickle.load(f)

    # Open the video file
    cap = cv2.VideoCapture(local_file_path)

    # Define IST timezone
    ist = pytz.timezone("Asia/Kolkata")

    while True:
        success, img = cap.read()

        # If the video reaches the end, reset to the beginning
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset the video to the beginning if end is reached
            success, img = cap.read()  # Try reading the first frame again
            if not success:
                break  # Exit if still unable to read the frame

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Preprocessing
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
        imgEdges = cv2.Canny(imgBlur, 50, 150)
        imgThreshold = cv2.adaptiveThreshold(
            imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16
        )
        imgCombined = cv2.bitwise_or(imgThreshold, imgEdges)
        kernel = np.ones((3, 3), np.uint8)
        imgDilate = cv2.dilate(imgCombined, kernel, iterations=1)
        imgOpening = cv2.morphologyEx(imgDilate, cv2.MORPH_OPEN, kernel)

        # Initialize counters for free and occupied slots
        free_slots = 0
        occupied_slots = 0

        # Analyze parking spots
        for pos in posList:
            x, y, w, h = pos["x"], pos["y"], pos["width"], pos["height"]
            imgCrop = imgOpening[y:y + h, x:x + w]
            count = cv2.countNonZero(imgCrop)
            area = w * h
            if count < area * 0.23:
                color = (0, 255, 0)  # Green for free
                free_slots += 1
            else:
                color = (0, 0, 255)  # Red for occupied
                occupied_slots += 1

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Display free and occupied slot counters on the video
        cv2.putText(
            img,
            f"Free Slots: {free_slots}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"Occupied Slots: {occupied_slots}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # Display current date and time in IST
        current_time = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            img,
            f"Date/Time: {current_time}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Encode frame
        ret, buffer = cv2.imencode(".jpg", img)
        frame = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()
    
@app.route("/process")
@login_required
def process_video():
    filename = session.get("filename", None)
    if not filename:
        return jsonify({"error": "No video uploaded. Please upload a video first."})
    return Response(generate_frames(filename), mimetype="multipart/x-mixed-replace; boundary=frame")

# Other functions like `generate_frames` remain unchanged

if __name__ == "__main__":
    app.run(debug=True)
