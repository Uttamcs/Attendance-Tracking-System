from io import BytesIO
import cv2
import os
from flask import Flask, request, render_template, redirect, send_file, session, url_for, jsonify
from geopy.distance import geodesic
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import qrcode
import pyzbar
import sys

try:
    from pyzbar.pyzbar import decode
    pyzbar_available = True
except ImportError:
    print("Warning: pyzbar not available. QR code scanning will be disabled.")
    pyzbar_available = False




# VARIABLES
MESSAGE = "WELCOME  Instruction: to register your attendance kindly click on 'a' on keyboard"

# Defining Flask App
app = Flask(__name__)

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(0)  # Adjust camera index as needed (1 or 0)
except:
    cap = cv2.VideoCapture(1)

# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

# Function to get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

# Function to extract the face from an image
def extract_faces(img):
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

# Function to identify face using ML model with confidence scores
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    labels = model.classes_
    proba = model.predict_proba(facearray)
    confidence_scores = np.max(proba, axis=1)
    predictions = model.predict(facearray)
    return predictions, confidence_scores

# Function to train the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

# Function to extract info from today's attendance file in attendance folder
# Function to extract info from today's attendance file in attendance folder
def extract_attendance():
    try:
        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')

        # Check if required columns are present
        if 'Name' not in df.columns or 'Roll' not in df.columns or 'Time' not in df.columns:
            print("Error: Required columns are missing from the CSV file.")
            return [], [], [], 0

        names = df['Name']
        rolls = df['Roll']
        times = df['Time']
        l = len(df)
        return names, rolls, times, l

    except FileNotFoundError:
        print(f"Attendance file for {datetoday} not found.")
        return [], [], [], 0

# In your 'add_attendance' function, ensure the file is being created correctly with the proper header
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    # Path to the CSV file
    csv_path = f'Attendance/Attendance-{datetoday}.csv'

    # Check if the CSV file exists, create it if it doesn't
    if not os.path.isfile(csv_path):
        with open(csv_path, 'w') as f:
            f.write('Name,Roll,Time\n')  # Writing header if file doesn't exist

    # Read existing data to prevent duplicate entries
    df = pd.read_csv(csv_path)
    if str(userid) not in df['Roll'].values:
        with open(csv_path, 'a') as f:
            f.write(f'{username},{userid},{current_time}\n')
        print(f"Attendance marked for {username}, ID: {userid}, at {current_time}")
    else:
        print(f"This user (ID: {userid}) has already marked attendance for today.")



# Routing Functions

# Main page
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2, mess=MESSAGE)

# Function to start attendance taking
# Function to start attendance taking
@app.route('/start', methods=['GET'])
def start():
    ATTENDENCE_MARKED = False
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        names, rolls, times, l = extract_attendance()
        MESSAGE = 'This face is not registered with us, kindly register yourself first'
        print("Face not in database, need to register")
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg,
                               datetoday2=datetoday2, mess=MESSAGE)

    cap = cv2.VideoCapture(0)  # Adjust camera index as needed (1 or 0)

    # Create a named window for fullscreen
    cv2.namedWindow("Attendance Check", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Attendance Check", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    ret = True
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            predictions, confidence_scores = identify_face(face.reshape(1, -1))
            identified_person = predictions[0]
            confidence = confidence_scores[0]
            cv2.putText(frame, f'{identified_person} (Confidence: {confidence:.2f})', (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)
            if cv2.waitKey(1) == ord('a'):
                add_attendance(identified_person)
                current_time_ = datetime.now().strftime("%H:%M:%S")
                print(f"Attendance marked for {identified_person}, at {current_time_}")
                ATTENDENCE_MARKED = True
                break
        if ATTENDENCE_MARKED:
            break

        # Display the resulting frame in the fullscreen window
        cv2.imshow('Attendance Check', frame)

        # Wait for the user to press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    MESSAGE = 'Attendance taken successfully'
    print("Attendance registered")
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2, mess=MESSAGE)


# Function to add a new user
@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        cap = cv2.VideoCapture(0)  # Adjust camera index as needed (1 or 0)
        i, j = 0, 0
        while 1:
            _, frame = cap.read()
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                            cv2.LINE_AA)
                if j % 10 == 0:
                    name = newusername + '_' + str(i) + '.jpg'
                    cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                    i += 1
                j += 1
            if j == 500:
                break
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        print('Training Model')
        train_model()
        names, rolls, times, l = extract_attendance()
        if totalreg() > 0:
            names, rolls, times, l = extract_attendance()
            MESSAGE = 'User added successfully'
            print("Message changed")
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                                   datetoday2=datetoday2, mess=MESSAGE)
        else:
            return redirect(url_for('home'))
    else:
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/start_qr', methods=['GET'])
def start_qr():
    cap = cv2.VideoCapture(0)
    ATTENDANCE_MARKED = False

    while True:
        ret, frame = cap.read()
        
        # Decode QR codes in the frame
        decoded_objects = decode(frame)
        
        for obj in decoded_objects:
            # Assuming the QR code contains the user's name
            name = obj.data.decode('utf-8')
            add_attendance(name)
            current_time_ = datetime.now().strftime("%H:%M:%S")
            print(f"Attendance marked for {name}, at {current_time_}")
            ATTENDANCE_MARKED = True
            break
        
        if ATTENDANCE_MARKED:
            break

        cv2.imshow('QR Code Scanner', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    names, rolls, times, l = extract_attendance()
    MESSAGE = 'Attendance taken successfully'
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2, mess=MESSAGE)

@app.route('/generate_qr/<name>')
def generate_qr(name):
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(name)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Save QR code to a BytesIO object
    img_io = BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png')


# Main function to run the Flask App
if __name__ == '__main__':
    app.run(debug=True, port=5000)
