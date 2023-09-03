import os
import cv2 as cv
import numpy as np

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

# Path of the directory containing the datasets
DIR = r'dataset'

# Load OpenCV's pre-trained Haar cascade for face detection
haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

features = []
labels = []

# Function to create and train the face recognizer
def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            # Load the image and convert to grayscale
            img_array = cv.imread(img_path)
            if img_array is None:
                print("Error loading image:", img_path)
                continue
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            # Detect faces in the image and extract regions of interest (ROIs)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            # Selecting the biggest face
            # (assuming that will be the inividual)
            if(len(faces_rect)  != 0):
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

                for (x, y, w, h) in faces_rect:
                    faces_roi = gray[y:y+h, x:x+w]
                    features.append(faces_roi)
                    labels.append(label)

# Call the function to create and train the face recognizer
create_train()
print("Training Done ------------------------>")

# Convert the features and labels to numpy arrays
features = np.array(features, dtype="object")
labels = np.array(labels)

# Create a LBPH face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the face recognizer using the features and labels
face_recognizer.train(features, labels)

# Save the trained face recognizer and the data
face_recognizer.save("face_trained.yml")
np.save("features.npy", features)
np.save("labels.npy", labels)
