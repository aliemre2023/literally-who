import cv2 as cv
import datetime
import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PySide6.QtGui import QPixmap


#### Load the Image ####
image_name = "me.jpg"
########################

img = cv.imread(f"you/{image_name}")

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

# Drawing rectangle and adding text on image
for (x, y, w, h) in faces_rect:
    # Extract the region of interest (ROI) for face recognition
    faces_roi = gray[y:y+h, x:x+w]

    # Perform face recognition to determine the person and confidence level
    label, confidence = face_recognizer.predict(faces_roi)
    print(f"Predicted Label: {people[label]} with Confidence: {confidence}")

    # Define font properties for displaying text
    font_scale = height / 800.0
    thickness = max(1, int(height / 300))
    text_size = cv.getTextSize(str(people[label]), cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    start_x = int((x + (w/2)) - text_size[0] / 2)

    # Display the name above and below the detected face
    cv.putText(img, str(people[label]), (start_x, y - int(height/50)), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
    cv.putText(img, str(people[label]), (start_x, y + h + text_size[1] + int(height/50)), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

    # Draw a rectangle around the detected face
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

# Save the output image to a 'saved' folder with a timestamp
current_time = datetime.datetime.now()
timestamp = current_time.strftime("%Y%m%d_%H%M%S")
output_path = f"saved/output_{timestamp}.jpg"
cv.imwrite(output_path, img)

# The Application
app = QApplication(sys.argv)
window = QMainWindow()
window.setGeometry(100, 100, 400, 600)

# Arrange image for App
desired_width = 300
ratio = height / width
desired_height = int(desired_width * ratio)
new_size = (desired_width, desired_height)
resized_image = cv.resize(img, new_size)
temp_output_path = "saved/resized_image.jpg"
cv.imwrite(temp_output_path, resized_image)

# Arrangements for displayed datas
img_label = QLabel(window)
img_label.setGeometry(50, 100, desired_width, desired_height)
text_2 = f"{people[label]}, loss:{int(confidence)}"
text_size_2 = cv.getTextSize(text_2, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
text_label = QLabel(window)
text_label.setGeometry(60, desired_height + 100, text_size_2[0], 30)

# Function to show arranged datas
def show_data():
    print("Button clicked")
    # Show image
    pixmap = QPixmap(temp_output_path)  
    img_label.setPixmap(pixmap)
    # Show text
    text_label.setText(text_2)
    
# Button for visualize the image
button = QPushButton("Run", window)
button.setGeometry(150, 35, 100, 30)
button.clicked.connect(show_data)

# Show all app
window.show()

# Closed the app
sys.exit(app.exec())