from flask import Flask, render_template , Response
from flask_pymongo import PyMongo  
from ultralytics import YOLO
import torch
import cv2
import numpy as np





app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')  


@app.route('/login')
def login():
    return render_template('login.html')  


@app.route('/register')
def register():
    return render_template('register.html') 


@app.route('/Fotp')
def Fotp():
    return render_template('Fotp.html')  


@app.route('/projectinfo')
def projectinfo():
    return render_template('projectinfo.html')  


#@app.route('/dashboard')
#def dashboard():
    #return render_template('dashboard.html')



 
model = YOLO('yolov8m.pt')  # Make sure the model is downloaded, or use the correct path

# Video capture function
def generate_video(video_file_path):
    cap = cv2.VideoCapture(video_file_path)
    
    
    # Ensure the video opens correctly
    if not cap.isOpened():
        raise ValueError("Error opening video stream or file")
    
    # Get the width and height of the video
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform YOLO prediction on the frame
        results = model.predict(frame)
        
        # Loop through results
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])  # Class ID
                conf = box.conf[0]  # Confidence score

                # Prepare label and draw bounding box
                label = f'Class:{cls} | Conf:{conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            break
        
        # Convert JPEG to bytes and yield to Response
        frame_data = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')

    # Release the video capture object
    cap.release()

# Route to serve video stream
@app.route('/video_feed')
def video_feed():
    # Provide the path to your video file here
    return Response(generate_video(r"C:\Users\arups\Downloads\videoplayback (1).mp4"),
                    mimetype='multipart/x-mixed-replace; boundary=frame') 












if __name__=="__main__":
    app.run(debug=True)
