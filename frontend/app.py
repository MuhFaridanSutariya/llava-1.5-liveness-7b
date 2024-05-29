import gradio as gr
import cv2
import numpy as np
from PIL import Image
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import os

from huggingface_hub import login
from peft import PeftModel, PeftConfig

login(token=os.environ["HF_TOKEN"])


# Check device availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configure 8-bit loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    # load_in_8bit_fp32_cpu_offload=True,
    # device_map="auto",
    # attn_implementation="flash_attention_2"
)

# Load model configurations
config = PeftConfig.from_pretrained("firqaaa/vsft-llava-1.5-7b-hf-liveness-trl")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
base_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",
                                                            torch_dtype=torch.float16,
                                                            low_cpu_mem_usage=True,
                                                            quantization_config=bnb_config)

model = PeftModel.from_pretrained(base_model, "firqaaa/vsft-llava-1.5-7b-hf-liveness-trl")
model.to(device)

def process_image(image):
    
    pil_image = Image.fromarray(image)
    
    prompt = """USER: <image>\nI ask you to be an liveness image annotator expert to determine if an image "Real" or "Spoof". 
                If an image is a "Spoof" define what kind of attack, is it spoofing attack that used Print(flat), Replay(monitor, laptop), or Mask(paper, crop-paper, silicone)?
                If an image is a "Real" or "Normal" return "No Attack". 
                Whether if an image is "Real" or "Spoof" give an explanation to this.
                Return your response using following format :

                Real/Spoof : 
                Attack Type :
                Explanation :\nASSISTANT:"""
    
    inputs = processor(prompt, images=pil_image, return_tensors="pt").to(device)
    
    output = model.generate(**inputs, max_new_tokens=300)
    
    decoded_output = processor.decode(output[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
    return image, decoded_output

def capture_and_process_image(webcam_image):
    captured_img, result = process_image(webcam_image)
    return captured_img, result

with gr.Blocks() as demo:
    with gr.Row():
        webcam_input = gr.Image(source="webcam", streaming=True, label="Webcam Input", height=483)
        captured_image = gr.Image(label="Captured Image", height=483)
    capture_button = gr.Button("Capture Image")
    result_output = gr.Textbox(label="Inference Result")

    capture_button.click(fn=capture_and_process_image, inputs=webcam_input, outputs=[captured_image, result_output])

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
