import cv2 as cv
import datetime

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

# Display the image with detected faces
cv.imshow("Detected Faces", img)

# Save the output image to a 'saved' folder with a timestamp
current_time = datetime.datetime.now()
timestamp = current_time.strftime("%Y%m%d_%H%M%S")
output_path = f"saved/output_{timestamp}.jpg"
cv.imwrite(output_path, img)

# Wait for a key press and then close the window
cv.waitKey(0)
