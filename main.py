import sys
import cv2 as cv
import datetime
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog
from PySide6.QtGui import QIcon, QPixmap, QPalette, QColor
import pygame

class LiterallyWho(QMainWindow):
    def __init__(self):
        super().__init__()

        self.selected_file = None
        self.human = None
        self.img_label = QLabel(self)
        self.text_label = QLabel(self)

        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 400, 600)
        self.setWindowTitle("Literally Who")
        self.setMinimumSize(400, 600)
        self.setMaximumSize(400, 600)

        # Set background image
        background_image = QPixmap("images/dolphin.png")
        background_label = QLabel(self)
        background_label.setPixmap(background_image)
        background_label.setGeometry(0, 0, 400, 600)

        # Create a QPalette and set the text color (foreground color)
        palette1 = QPalette()
        text_color = QColor(255, 33, 255)
        palette1.setColor(QPalette.WindowText, text_color)
        palette2 = QPalette()
        text_color = QColor(0, 255, 255)
        palette2.setColor(QPalette.WindowText, text_color)

        # Button for file selection
        button_opn = QPushButton("Select a File", self)
        button_opn.setGeometry(150, 15, 100, 30)
        button_opn.setStyleSheet("background-color: pink;")
        button_opn.clicked.connect(self.open_file_dialog)

        # Label to show file name
        self.path_label = QLabel(self)
        self.path_label.setGeometry(60, 90, 280, 30)
        self.path_label.setPalette(palette2)

        # Create labels for displaying data
        self.img_label = QLabel(self)
        self.img_label.setGeometry(50, 120, 300, 400)
        self.text_label = QLabel(self)
        self.text_label.setGeometry(-30, 550, 460, 30)
        self.text_label.setPalette(palette1)

        # Button for visualize the image
        button_run = QPushButton("Run", self)
        button_run.setGeometry(150, 50, 100, 30)
        button_run.setStyleSheet("background-color: green;")
        button_run.clicked.connect(self.show_data)
        button_run.clicked.connect(self.play_sound)

    def open_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_dialog = QFileDialog()
        file_dialog.setOptions(options)
        file_dialog.setNameFilter("Image Files (*.jpg *.jpeg *.png *.gif);;All Files (*)")

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.selected_file = selected_files[0]
                print("Selected file:", self.selected_file)

                # Label to show image path
                ways = self.selected_file.split("/")
                outp = ways[-1]
                self.path_label.setText("Selected Image: " + outp)
                self.path_label.show()

    def show_data(self):
        if self.selected_file is None:
            return
        
        img = cv.imread(self.selected_file)

        # Load OpenCV's pre-trained Haar cascade for face detection
        haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # List of recognized individuals
        people = ["Ryan Gosling", 
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
                "Travis Bickle (Taxi Driver)",
                "Obama",
                "Jesse Pinkman",
                "Nejat Isler",
                "Issiz Adam Alper",
                "Cayci Huseyin"]

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

        if(hB == 0):
            print("No face detected !")
        else:
            # Drawing rectangle and adding text on image
            for (x, y, w, h) in faces_rect:
                # Extract the region of interest (ROI) for face recognition
                faces_roi = gray[y:y+h, x:x+w]

                # Perform face recognition to determine the person and confidence level
                label, confidence = face_recognizer.predict(faces_roi)
                print(f"Predicted Label: {people[label]} with Confidence: {confidence}")
                
                self.human = people[label]

                # Define font properties for displaying text
                text_1 = str(people[label])
                font_scale = height / 800.0
                thickness = max(2, int(height / 300))
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
            self.img_label.setPixmap(pixmap)
            # Show text on app
            self.text_label.setStyleSheet("border: 2px solid #ff33ff")
            text_2 = f"{people[label]} - {people[label]} - {people[label]} - {people[label]} - {people[label]} - {people[label]} - {people[label]} - {people[label]} - {people[label]} - {people[label]}"
            self.text_label.setText(text_2)
        
    def play_sound(self):
        pygame.init()
        # base music
        pygame.mixer.music.load("audios/The-Cruel-Angel's-Thesis.mp3") 
        if self.human == "Burhan Altintop":
            pygame.mixer.music.load("audios/burhan-altintop.mp3")
        elif(self.human == "Patrick Bateman"):
            pygame.mixer.music.load("audios/patrick-bateman.mp3")
        elif(self.human == "Ryan Gosling"):
            pygame.mixer.music.load("audios/ryan-gosling.mp3")
        elif(self.human == "Recep Ivedik"):
            pygame.mixer.music.load("audios/recep-ivedik.mp3")
        elif(self.human == "Tyler Durden"):
            pygame.mixer.music.load("audios/tyler-durden.mp3")       
        elif(self.human == "The Narrator"):
            pygame.mixer.music.load("audios/tyler-durden.mp3")   
        elif(self.human == "Joker (Joaquin Phoenix)"):
            pygame.mixer.music.load("audios/joker-joaquin-phoenix.mp3")   
        elif(self.human == "Joker (Heath Ledger)"):
            pygame.mixer.music.load("audios/joker-heath-ledger.mp3")
        elif(self.human == "Batman (The Batman)"):
            pygame.mixer.music.load("audios/batman-the-batman.mp3")
        elif(self.human == "Jesse Pinkman"):
            pygame.mixer.music.load("audios/jesse-pinkman.mp3")
        elif(self.human == "Walter White"):
            pygame.mixer.music.load("audios/walter-white.mp3")
        elif(self.human == "Saul Goodman"):
            pygame.mixer.music.load("audios/saul-goodman.mp3")
        elif(self.human == "Obama"):
            pygame.mixer.music.load("audios/obama.mp3")
        elif(self.human == "Travis Bickle (Taxi Driver)"):
            pygame.mixer.music.load("audios/travis-bickle-taxi-driver.mp3")
        elif(self.human == "Issiz Adam Alper"):
            pygame.mixer.music.load("audios/issiz-adam-alper.mp3")
        elif(self.human == "Nejat Isler"):
            pygame.mixer.music.load("audios/nejat-isler.mp3")
        elif(self.human == "Cayci Huseyin"):
            pygame.mixer.music.load("audios/cayci-huseyin.mp3")
        
        else:
            print("Music not founded.")
        pygame.mixer.music.play()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("images/icon.ico"))
    mainWindow = LiterallyWho()
    mainWindow.show()
    sys.exit(app.exec())