import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to load image
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Function to detect faces
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

# Function to draw rectangles around faces
def draw_faces(image, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image

# Streamlit app
st.title("Testing upload detect muka messi")

st.write("Upload messi the goat")

# File uploader
image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if image_file is not None:
    # Load image
    img = load_image(image_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert PIL image to OpenCV format
    img_array = np.array(img)
    image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Detect faces
    faces = detect_faces(image)
    
    if len(faces) > 0:
        st.write(f"Found {len(faces)} face(s)")
        # Draw faces
        image_with_faces = draw_faces(image, faces)
        # Convert back to RGB format for displaying in Streamlit
        image_with_faces = cv2.cvtColor(image_with_faces, cv2.COLOR_BGR2RGB)
        st.image(image_with_faces, caption="Image with Detected Faces", use_column_width=True)
    else:
        st.write("No faces found")
