import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from peft import PeftModel, PeftConfig
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import torch
from PIL import Image
import cv2

# Check device availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configure 8-bit loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit_fp32_cpu_offload=True,
    device_map="auto",
    attn_implementation="flash_attention_2"
)

# Load model configurations
config = PeftConfig.from_pretrained("firqaaa/vsft-llava-1.5-7b-hf-liveness-trl")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
base_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",
                                                            torch_dtype=torch.float16,
                                                            quantization_config=bnb_config)

model = PeftModel.from_pretrained(base_model, "firqaaa/vsft-llava-1.5-7b-hf-liveness-trl")
model.to(device)

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame):
        self.frame = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2RGB)
        return av.VideoFrame.from_ndarray(self.frame, format="bgr24")

st.title("WebRTC Camera Capture")
st.write("Capture an image from your camera.")

webrtc_ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor)

if webrtc_ctx.video_processor:
    if st.button("Capture"):
        img = webrtc_ctx.video_processor.frame
        if img is not None:
            st.image(img, caption='Captured Image.', use_column_width=True)
            
            image = Image.fromarray(img)
            prompt = """USER: <image>\nI ask you to be an liveness image annotator expert to determine if an image "Real" or "Spoof". 
                        If an image is a "Spoof" define what kind of attack, is it spoofing attack that used Print(flat), Replay(monitor, laptop), or Mask(paper, crop-paper, silicone)?
                        If an image is a "Real" or "Normal" return "No Attack". 
                        Whether if an image is "Real" or "Spoof" give an explanation to this.
                        Return your response using following format :

                        Real/Spoof : 
                        Attack Type :
                        Explanation :\nASSISTANT:"""
            
            # Prepare inputs and move to device
            inputs = processor(prompt, images=image, return_tensors="pt").to(device)
            
            # Generate output
            output = model.generate(**inputs, max_new_tokens=300)
            
            # Decode and display the output
            decoded_output = processor.decode(output[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
            st.write("Inference Result:")
            st.write(decoded_output)
