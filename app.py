from flask import Flask, render_template, request, Response
import os
import shutil
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import smtplib  # NEW: For email alerts
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from utils import predict_violence, predict_violence_frames

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
FRAME_FOLDER = 'static/frames'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# NEW: Email Configuration - UPDATE THESE WITH YOUR DETAILS
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SENDER_EMAIL = 'ubaleritesh57@gmail.com'  # Your sender email
SENDER_PASSWORD = 'nllt tzlv mlwp ohhn'   # Gmail App Password (not main password)
RECIPIENT_EMAIL = 'ubaleritesh6062@gmail.com'  # Office email for mobile alerts
ALERT_SUBJECT = '🚨 Violence Detected in Live Feed'
ALERT_BODY = 'Violence detected with high confidence in the live camera feed. Check immediately: http://127.0.0.1:5000/live'

# NEW: Email alert function
def send_violence_alert():
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = ALERT_SUBJECT
    msg.attach(MIMEText(ALERT_BODY, 'plain'))
    
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        print("✅ Violence alert email sent successfully!")
        return True
    except Exception as e:
        print(f"❌ Email error: {e}")
        return False

buffer = []
smooth_prob = 0.0
frame_count = 0
camera = None
motion_threshold = 0.05
violence_threshold = 0.80  # FIXED: Higher threshold
last_alert_time = type('obj', (), {'time': 0})()  # NEW: Cooldown timer for alerts

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    accuracy = None
    filename = None
    frames_list = []
    graph_path = None
    
    if request.method == 'POST':
        file = request.files['file']
        if file.filename:
            filename = file.filename
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            
            shutil.rmtree(FRAME_FOLDER)
            os.makedirs(FRAME_FOLDER)
            
            result, prob = predict_violence(path, FRAME_FOLDER)
            accuracy = round(prob * 100, 2)
            
            for img in sorted(os.listdir(FRAME_FOLDER)):
                frames_list.append(os.path.join(FRAME_FOLDER, img))
            
            plt.figure(figsize=(8, 6))
            plt.bar(['Violent', 'Non-Violent'], [accuracy, 100-accuracy], 
                   color=['red', 'green'], alpha=0.7)
            plt.ylabel('Probability')
            plt.title('Violence Detection Result')
            graph_path = 'static/graph.png'
            plt.savefig(graph_path, bbox_inches='tight')
            plt.close()
    
    return render_template('index.html', 
                         result=result, 
                         accuracy=accuracy, 
                         filename=filename, 
                         frames=frames_list, 
                         graph=graph_path)

def generate_frames():
    global buffer, smooth_prob, frame_count, camera, last_alert_time
    
    if camera is None:
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        time.sleep(2.0)
        print("✅ Live Violence Detection Started!")
    
    while True:
        success, frame = camera.read()
        if not success:
            time.sleep(0.01)
            continue
        
        frame_count += 1
        frame_display = cv2.resize(frame, (640, 480))
        
        # Initialize camera - Your Camera 0 confirmed working...
        if frame_count % 15 == 0:  # Process every 15th frame
            try:
                resized = cv2.resize(frame, (128, 128))
                normalized = resized.astype(np.float32) / 255.0
                buffer.append(normalized)
                
                if len(buffer) > 20:
                    buffer.pop(0)  # FIXED: Smart motion + violence detection...
                
                if len(buffer) == 20:
                    frames_array = np.array(buffer)  # Only predict when we have enough frames...
                    
                    motion = np.mean(np.std(frames_array, axis=0))  # Motion magnitude
                    
                    if motion > motion_threshold:  # Only process if there's movement
                        label, prob = predict_violence_frames(frames_array)  # KEY FIX: Motion detection first...
                        
                        smooth_prob = 0.85 * smooth_prob + 0.15 * prob
                    else:  # Slower smoothing for stability...
                        smooth_prob = 0.95 * smooth_prob + 0.05 * 0.0
                        
            except Exception as e:
                print(f'Prediction error: {e}')
                pass  # No motion = non-violent...
        
        final_label = "VIOLENT" if smooth_prob > violence_threshold else "Safe"
        confidence = round(smooth_prob * 100, 1)  # FIXED: Higher threshold + clear labels...
        
        color = (0, 0, 255) if smooth_prob > violence_threshold else (0, 255, 0)
        thickness = 3 if smooth_prob > violence_threshold else 2  # Color coding...
        
        cv2.putText(frame_display, final_label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, thickness)
        cv2.putText(frame_display, f'Confidence: {confidence}%', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_display, f'Motion: {motion_threshold:.2f}', (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)  # Multiple text overlays...
        
        # NEW: Violence Alert with Email + Visual Warning
        if smooth_prob > violence_threshold:
            cv2.rectangle(frame_display, (0, 0), (640, 480), (0, 0, 255), 5)
            
            # Send email alert (1 minute cooldown to avoid spam)
            current_time = time.time()
            if current_time - last_alert_time.time > 60:
                send_violence_alert()
                last_alert_time.time = current_time
        
        ret, buffer_img = cv2.imencode('.jpg', frame_display, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer_img.tobytes() + b'\r\n')

@app.route('/live')
def live():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("🚀 Violence Detection System Starting...")
    print("📱 Home: http://127.0.0.1:5000")
    print("🔴 Live: http://127.0.0.1:5000/live")
    
    try:
        app.run(debug=False, use_reloader=False, threaded=True, host='127.0.0.1', port=5000)
    finally:
        if camera:
            camera.release()
        cv2.destroyAllWindows()
        print("🛑 Server stopped")
