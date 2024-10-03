import sys
import os
import uuid
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QFileDialog, QWidget, QTextEdit
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from ultralytics import YOLO
import cv2
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Object Detection")
        self.setGeometry(100, 100, 800, 600)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Create a label for displaying the image
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)

        # Create a button for uploading files
        self.upload_button = QPushButton("Upload Image/Video", self)
        self.upload_button.clicked.connect(self.upload_file)
        layout.addWidget(self.upload_button)

        # Create a text edit widget for displaying output
        self.output_text = QTextEdit(self)
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)

        # Set the central widget
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Initialize the model
        self.model = YOLO("models/best2.pt")  # pretrained YOLOv8n model

    def upload_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Images (*.png *.jpg *.jpeg);;Videos (*.mp4 *.avi)", options=options)

        if file_name:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.process_image(file_name)
            elif file_name.lower().endswith(('.mp4', '.avi')):
                self.process_video(file_name)

    def process_image(self, file_path):
        img = cv2.imread(file_path)
        results = self.model(img, stream=True)

        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                filtered_boxes = [box for box in boxes if box.cls == 0]

                if len(filtered_boxes) > 0:
                    result_filtered = result.clone()
                    result_filtered.boxes = filtered_boxes
                    annotated_img = result_filtered.plot()

                    # Display the annotated image
                    pixmap = QPixmap.fromImage(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
                    self.image_label.setPixmap(pixmap.scaled(600, 400, Qt.AspectRatioMode.KeepAspectRatio))

                    # Display the output text
                    output_text = f"Detected {len(filtered_boxes)} objects."
                    self.output_text.append(output_text)

    def process_video(self, file_path):
        cap = cv2.VideoCapture(file_path)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, stream=True)

            for result in results:
                boxes = result.boxes
                if len(boxes) > 0:
                    filtered_boxes = [box for box in boxes if box.cls == 0]

                    if len(filtered_boxes) > 0:
                        result_filtered = result.clone()
                        result_filtered.boxes = filtered_boxes
                        annotated_frame = result_filtered.plot()

                        # Display the annotated frame
                        cv2.imshow('Annotated Frame', annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())