import streamlit as st
import cv2
import numpy as np

# Function to open the webcam and stream frames
def open_webcam():
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    frame_placeholder = st.empty()  # Placeholder for displaying frames

    while st.session_state["webcam_active"]:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame.")
            break
        
        # Convert the frame to RGB format for displaying in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame in Streamlit
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

    # Release the webcam
    cap.release()

# Initialize session state for webcam activity
if "webcam_active" not in st.session_state:
    st.session_state["webcam_active"] = False

# Title of the Streamlit app
st.title("Tes Webcam Wajah Tampan")

# Instructions
st.write("gaada stop button nya, jadi kalo mau stop, refresh aja page nya.")

# Start/Stop buttons for the webcam
if st.button("Start Webcam"):
    st.session_state["webcam_active"] = True
    open_webcam()