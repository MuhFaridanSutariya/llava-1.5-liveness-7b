import gradio as gr
from vecstore import register_face

with gr.Blocks() as demo:
    with gr.Row():
        webcam_input = gr.Image(source="webcam", streaming=True, label="Webcam Input", height=483)
        captured_image = gr.Image(label="Captured Image", height=483)
    user_name = gr.Textbox(label="User Name")
    register_button = gr.Button("Register Face")
    if register_button:
        result_output = gr.Textbox(label="Result", visible=True)

    def register_and_show_result(webcam_image, user_name):
        image, result = register_face(webcam_image, user_name)
        result_output.update(visible=True)
        return image, result

    register_button.click(fn=register_and_show_result, inputs=[webcam_input, user_name], outputs=[captured_image, result_output])

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
