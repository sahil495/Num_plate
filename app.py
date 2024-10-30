

import streamlit as st
import cv2
import pytesseract
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os

# Load the YOLO model
model = YOLO('best.pt')  # Ensure best.pt is in the same directory as your app

# Configure Tesseract OCR path based on the environment
if os.name == 'nt':  # For Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:  # For Linux/Streamlit Cloud
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

st.title("Number Plate Detection")

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the image file
    image = Image.open(uploaded_file)
    original_image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert to OpenCV format

    # Variable to store detected number plates
    detected_plates = []

    # Perform YOLO detection on the original image
    results = model.predict(original_image_cv, conf=0.25)  # Adjust confidence as needed

    # Process detection results
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()  # Bounding box coordinates
        label = results[0].names[int(box.cls)]  # Class name
        confidence = box.conf[0]  # Confidence score

        # Draw rectangle and label on the original image
        cv2.rectangle(original_image_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(original_image_cv, f'{label}: {confidence:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Extract the number plate region
        number_plate_img = original_image_cv[y1:y2, x1:x2]

        if number_plate_img.size != 0:
            # Convert to grayscale and apply thresholding
            number_plate_gray = cv2.cvtColor(number_plate_img, cv2.COLOR_BGR2GRAY)
            _, number_plate_thresh = cv2.threshold(number_plate_gray, 150, 255, cv2.THRESH_BINARY)

            # Perform OCR on the thresholded image
            ocr_result = pytesseract.image_to_string(number_plate_thresh, config='--psm 8').strip()
            detected_plates.append(ocr_result)  # Store the detected number plate

            if ocr_result:
                cv2.putText(original_image_cv, ocr_result, (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Convert the original image back to RGB for Streamlit
    output_image = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2RGB)
    
    # Show detected number plates separately
    if detected_plates:
        st.subheader("Detected Plate Number:")
        for plate in detected_plates:
            st.write(plate)
    else:
        st.write("No number plates detected.")

    # Display the processed image with a fixed width of 600 pixels
    st.image(output_image, caption='Processed Image', width=600)

