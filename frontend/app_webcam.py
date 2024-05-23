import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame):
        # Convert the frame to RGB using OpenCV
        self.frame = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2RGB)
        return av.VideoFrame.from_ndarray(self.frame, format="rgb24")

st.title("WebRTC Camera Capture")
st.write("Capture an image from your camera.")

webrtc_ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor)

if webrtc_ctx.video_processor:
    if st.button("Capture"):
        img = webrtc_ctx.video_processor.frame
        if img is not None:
            st.image(img, caption='Captured Image.', use_column_width=True)
            st.write("")
