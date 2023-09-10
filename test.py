import cv2 as cv
import datetime
import time
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog
from PySide6.QtGui import QPixmap, QIcon, QPalette, QColor
import pygame

# THE APPLICATION
app = QApplication(sys.argv)
window = QMainWindow()
window.setGeometry(100, 100, 400, 600)
window.setWindowTitle("Literally Who")
app.setWindowIcon(QIcon("images/icon.ico"))
window.setMinimumSize(400, 600)
window.setMaximumSize(400, 600)

# Set background image
background_image = QPixmap("images/dolphin.png")
background_label = QLabel(window)
background_label.setPixmap(background_image)
background_label.setGeometry(0, 0, 400, 600)

# Create a QPalette and set the text color (foreground color)
palette1 = QPalette()
text_color = QColor(255, 33, 255)  
palette1.setColor(QPalette.WindowText, text_color)
palette2 = QPalette()
text_color = QColor(00, 255, 255)  
palette2.setColor(QPalette.WindowText, text_color)

# Button for file selctation
button_opn = QPushButton("Select a File", window)
button_opn.setGeometry(150, 15, 100, 30)
button_opn.setStyleSheet("background-color: pink;")
# Label to show file name
path_label = QLabel(window)
path_label.setGeometry(60, 90, 280, 30)
path_label.setPalette(palette2)

selected_file = None
def open_file_dialog():
    global selected_file
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    file_dialog = QFileDialog()
    file_dialog.setOptions(options)
    file_dialog.setNameFilter("Image Files (*.jpg *.jpeg *.png *.gif);;All Files (*)")

    if file_dialog.exec():
        selected_files = file_dialog.selectedFiles()
        if selected_files:
            selected_file = selected_files[0]
            print("Selected file:", selected_file)

            # Label to show image path
            ways = selected_file.split("/")
            outp = ways[-1]
            text_size = cv.getTextSize(str(outp), cv.FONT_HERSHEY_SIMPLEX, fontScale=2.0, thickness=2)[0]
            #path_label.setGeometry(200 - text_size[0]/2, 90, 100, 30)
            #time.sleep(1)
            path_label.setText("Selected Image: " + outp)
            path_label.show()

button_opn.clicked.connect(open_file_dialog)

# Create labels for displaying data
img_label = QLabel(window)
img_label.setGeometry(50, 120, 300, 400)
text_label = QLabel(window)
text_label.setGeometry(-30, 550, 460, 30)
text_label.setPalette(palette1)

human = None
def show_data():
    if selected_file is None:
        return

    img = cv.imread(selected_file)

    # Load OpenCV's pre-trained Haar cascade for face detection
    haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # List of recognized individuals
    people = ["The Driver (Drive)", 
            "Patrick Bateman", 
            "Burhan Altintop", 
            "Recep Ivedik", 
            "Tyler Durden",
            "The Narrator",
            "Batman (The Batman)",
            "Walter White",
            "Joker (Joaquin Phoenix)",
            "Joker (Heath Ledger)",
            "Saul Goodman",
            "Joe (Blade Runner 2049)",
            "Travis Bickle (Taxi Driver)",
            "Obama",
            "Jesse Pinkman"]

    # Load a trained face recognizer model
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.read("face_trained.yml")

    # Get the height and width of the image
    height, width = img.shape[:2]

    # Convert the image to grayscale for better analysis
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect faces using the Haar cascade
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # Select the biggest face
    biggest_rect = -1
    xB = 0
    yB = 0
    wB = 0
    hB = 0
    for(x, y, w, h) in faces_rect:
        new_rect = abs(w)
        if(biggest_rect < new_rect):
            biggest_rect = new_rect
            xB = x
            yB = y
            wB = w
            hB = h

    # Indexes of biggest face
    faces_rect = [[xB, yB, wB, hB]]

    # Drawing rectangle and adding text on image
    for (x, y, w, h) in faces_rect:
        # Extract the region of interest (ROI) for face recognition
        faces_roi = gray[y:y+h, x:x+w]

        # Perform face recognition to determine the person and confidence level
        label, confidence = face_recognizer.predict(faces_roi)
        print(f"Predicted Label: {people[label]} with Confidence: {confidence}")
        global human
        human = people[label]

        # Define font properties for displaying text
        text_1 = str(people[label])
        font_scale = height / 800.0
        thickness = max(1, int(height / 300))
        text_size = cv.getTextSize(text_1, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]   
        start_x = int((x + (w/2)) - text_size[0] / 2)
        
        # Display the name above and below the detected face
        cv.putText(img, text_1, (start_x, y - int(height/50)), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        
        # Draw a rectangle around the detected face
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    # Save the output image to a 'saved' folder with a timestamp
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    output_path = f"saved/output_{timestamp}.jpg"
    cv.imwrite(output_path, img)

    # Arrange image for App
    desired_width = 300
    ratio = height / width
    desired_height = int(desired_width * ratio)
    new_size = (desired_width, desired_height)
    resized_image = cv.resize(img, new_size)
    temp_output_path = "saved/temp_image.jpg"
    cv.imwrite(temp_output_path, resized_image)

    # Show image on app
    temp_output_path = "saved/temp_image.jpg"
    pixmap = QPixmap(temp_output_path)
    img_label.setPixmap(pixmap)
    # Show text on app
    text_label.setStyleSheet("border: 2px solid #ff33ff")
    text_2 = f"{people[label]} - {people[label]} - {people[label]} - {people[label]} - {people[label]} - {people[label]} - {people[label]}"
    text_label.setText(text_2)

pygame.init()    
def play_sound():
    global human
    # Audio playing
    if(human == "Burhan Altintop"):
        pygame.mixer.music.load("audios/burhan-altintop.mp3")
        pygame.mixer.music.play()
    elif(human == "Patrick Bateman"):
        pygame.mixer.music.load("audios/patrick-bateman.mp3")
        pygame.mixer.music.play()
    elif(human == "The Driver (Drive)"):
        pygame.mixer.music.load("audios/the-driver-drive.mp3")
        pygame.mixer.music.play()
    elif(human == "Recep Ivedik"):
        pygame.mixer.music.load("audios/recep-ivedik.mp3")
        pygame.mixer.music.play()
    else:
        print("Music not added yet.")

# Button for visualize the image
button_run = QPushButton("Run", window)
button_run.setGeometry(150, 50, 100, 30)
button_run.setStyleSheet("background-color: green;")
button_run.clicked.connect(show_data)
button_run.clicked.connect(play_sound)

# Show all app
window.show()

# Closed the app
sys.exit(app.exec())