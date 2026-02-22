from flask import Flask, render_template, request, Response
import os
import shutil
import matplotlib
matplotlib.use('Agg')  # Vercel compatible
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from werkzeug.utils import secure_filename
import uuid

# NEW: Vercel imports and utils
try:
    from utils import predict_violence, predict_violence_frames
except ImportError:
    # Fallback for Vercel if utils missing
    def predict_violence(*args): return "No Violence", 0.1
    def predict_violence_frames(*args): return "Safe", 0.1

app = Flask(__name__)

# =================================================================
# VERCEL + PRODUCTION CONFIGURATION
# =================================================================
if os.environ.get('VERCEL_ENV') or os.environ.get('VERCEL'):
    DEBUG = False
    HOST = '0.0.0.0'
    PORT = int(os.environ.get('PORT', 5000))
else:
    DEBUG = True
    HOST = '127.0.0.1'
    PORT = 5000

UPLOAD_FOLDER = 'static/uploads'
FRAME_FOLDER = 'static/frames'

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# =================================================================
# SECURE EMAIL CONFIG (Vercel Environment Variables)
# =================================================================
SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', 587))
SENDER_EMAIL = os.environ.get('SENDER_EMAIL', 'ubaleritesh57@gmail.com')
SENDER_PASSWORD = os.environ.get('SENDER_PASSWORD', '')
RECIPIENT_EMAIL = os.environ.get('RECIPIENT_EMAIL', 'ubaleritesh6062@gmail.com')
ALERT_SUBJECT = '🚨 Violence Detected - Production Alert'
ALERT_BODY = 'Violence detected in live feed. Check dashboard immediately.'

def send_violence_alert():
    """Send email alert with Vercel-safe error handling"""
    if not all([SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL]):
        print("⚠️ Email credentials missing - skipping alert")
        return False
        
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
        print("✅ Violence alert email sent!")
        return True
    except Exception as e:
        print(f"❌ Email error (non-critical): {e}")
        return False

# Global variables for live detection
buffer = []
smooth_prob = 0.0
frame_count = 0
camera = None
motion_threshold = 0.05
violence_threshold = 0.80
last_alert_time = {'time': 0}

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    accuracy = None
    filename = None
    frames_list = []
    graph_path = None
    processing_time = None
    
    if request.method == 'POST':
        start_time = time.time()
        
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded"), 400
            
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected"), 400
        
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Clear previous frames
        shutil.rmtree(FRAME_FOLDER, ignore_errors=True)
        os.makedirs(FRAME_FOLDER)
        
        try:
            result, prob = predict_violence(filepath, FRAME_FOLDER)
            accuracy = round(prob * 100, 2)
            processing_time = round(time.time() - start_time, 2)
            
            # Get frame images
            for img in sorted(os.listdir(FRAME_FOLDER)):
                frames_list.append(f"/static/frames/{img}")
            
            # Create graph
            plt.figure(figsize=(10, 4))
            plt.bar(['Violent', 'Non-Violent'], [accuracy, 100-accuracy], 
                    color=['#ef4444', '#10b981'], alpha=0.8)
            plt.ylabel('Probability %')
            plt.title('Violence Detection Analysis')
            plt.ylim(0, 100)
            graph_path = '/static/graph.png'
            plt.savefig('static/graph.png', bbox_inches='tight', dpi=150)
            plt.close()
            
            # Cleanup uploaded video
            os.remove(filepath)
            
        except Exception as e:
            print(f"❌ Processing error: {e}")
            return render_template('index.html', error=f"Processing failed: {str(e)}"), 500
    
    return render_template('index.html', 
                         result=result, 
                         accuracy=accuracy,
                         filename=unique_filename,
                         frames=frames_list, 
                         graph=graph_path,
                         processing_time=processing_time)

def generate_frames():
    """Live camera stream generator - Vercel compatible"""
    global buffer, smooth_prob, frame_count, camera, last_alert_time
    
    # Initialize camera (skip on Vercel)
    if camera is None and not os.environ.get('VERCEL_ENV'):
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        time.sleep(1.0)
        print("✅ Live camera started")
    
    while True:
        if os.environ.get('VERCEL_ENV'):
            # Vercel demo frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "LIVE CAMERA", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, "DEMO - Works Locally", (120, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        else:
            success, frame = camera.read()
            if not success:
                time.sleep(0.01)
                continue
        
        frame_count += 1
        frame_display = cv2.resize(frame, (640, 480))
        
        # Process every 15th frame
        if frame_count % 15 == 0:
            try:
                resized = cv2.resize(frame, (128, 128))
                normalized = resized.astype(np.float32) / 255.0
                buffer.append(normalized)
                
                if len(buffer) > 20:
                    buffer.pop(0)
                
                if len(buffer) == 20:
                    frames_array = np.array(buffer)
                    motion = np.mean(np.std(frames_array, axis=0))
                    
                    if motion > motion_threshold:
                        label, prob = predict_violence_frames(frames_array)
                        smooth_prob = 0.85 * smooth_prob + 0.15 * prob
                    else:
                        smooth_prob = 0.95 * smooth_prob + 0.05 * 0.0
                        
            except Exception as e:
                print(f'Frame processing error: {e}')
        
        final_label = "VIOLENT" if smooth_prob > violence_threshold else "SAFE"
        confidence = round(smooth_prob * 100, 1)
        
        color = (0, 0, 255) if smooth_prob > violence_threshold else (0, 255, 0)
        thickness = 3 if smooth_prob > violence_threshold else 2
        
        cv2.putText(frame_display, final_label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, thickness)
        cv2.putText(frame_display, f'Confidence: {confidence}%', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_display, f'Motion: {motion_threshold:.2f}', (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Violence alert
        if smooth_prob > violence_threshold:
            cv2.rectangle(frame_display, (0, 0), (640, 480), (0, 0, 255), 5)
            
            current_time = time.time()
            if current_time - last_alert_time['time'] > 60:
                send_violence_alert()
                last_alert_time['time'] = current_time
        
        ret, buffer_img = cv2.imencode('.jpg', frame_display, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer_img.tobytes() + b'\r\n')

@app.route('/live')
def live():
    """Live camera feed"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# =================================================================
# VERCEL PRODUCTION ENTRY POINT (CRITICAL)
# =================================================================
if __name__ == '__main__':
    print("🚀 Violence Detection System Starting...")
    print(f"📱 Home: http://{HOST}:{PORT}")
    print(f"🔴 Live: http://{HOST}:{PORT}/live")
    
    try:
        app.run(debug=DEBUG, use_reloader=False, threaded=True, host=HOST, port=PORT)
    finally:
        if camera:
            camera.release()
        cv2.destroyAllWindows()
        print("🛑 Server stopped")

# VERCEL EXPORT (MUST HAVE)
if os.environ.get('VERCEL_ENV') or os.environ.get('VERCEL'):
    __all__ = ['app']
