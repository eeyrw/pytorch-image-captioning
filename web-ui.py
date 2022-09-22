import numpy as np
import gradio as gr
from inference import PerapreModel,evaluate_single


def genTxtFromImage(x):
    text = evaluate_single(x,tokenInfo, encoder, decoder, max_len, device)
    return text

encoder, decoder, tokenInfo, max_len, device = PerapreModel()
     # Evaluate model performance on subsets
    

with gr.Blocks() as demo:
    gr.Markdown("Image caption demo")
    with gr.TabItem("Image"):
        with gr.Row():
            image_input = gr.Image()
            text_output = gr.Textbox()
        image_button = gr.Button("Caption it")

    image_button.click(genTxtFromImage, inputs=image_input, outputs=image_input)
    
demo.launch()