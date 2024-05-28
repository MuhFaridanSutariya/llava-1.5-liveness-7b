import gradio as gr
import cv2
import numpy as np
from PIL import Image
import tempfile


def show(img):
    return np.array(img)

with gr.Blocks() as demo:
    with gr.Row():
        video_input = gr.Image(source="webcam", streaming=True, height=483)
        captured_image = gr.Image(label="Captured Image", height=483)
    capture_button = gr.Button("Capture Image")       
    capture_button.click(fn=show, inputs=video_input, outputs=captured_image)

if __name__ == "__main__":
    demo.queue().launch()
