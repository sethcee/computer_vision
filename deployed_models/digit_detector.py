# get packages
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import io
import base64
import json
from PIL import Image
import requests

# lambda endpoint url
url = "https://w4rphwvjdf3v44f34tfgzd7qwq0pezcg.lambda-url.us-east-1.on.aws/"

# configure the application layout
st.set_page_config(page_title = "Digit Detector", layout = "centered")
st.title("Handwritten Digit Detector")
st.write("Draw a single digit on the canvas and press the infer digit button.")
column1, column2, = st.columns((1,1))

# set state session for reset
if "reset_key" not in st.session_state :
    st.session_state.reset_key = 0

# function to force canvas reset
def reset_canvas() :
    st.session_state.reset_key += 1

# create the layout
with column1 :
    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 1)", 
        stroke_width=20,
        stroke_color="#FFFFFF", 
        background_color="#000000",
        update_streamlit=True, 
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.reset_key}",
    )

with column2 :
    st.write("Controls")

    infer_button = st.button("Infer digit", use_container_width = True)
    st.button("Reset", on_click = reset_canvas, use_container_width = True)
    st.divider()

    st.write("### Output")
    output_display = st.empty()

if infer_button :
    if canvas.image_data is not None :
        image = canvas.image_data

        if np.max(image[:, :, :3]) > 10:
            
            image = Image.fromarray(image.astype('uint8'))
            image = image.convert("L")

            buffer = io.BytesIO()
            image.save(buffer, format = "PNG")

            image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            payload = {
                "data" : image_base64
                }

            # lambda cold start means we might need to try twice
            for i in range(2) : 
                response = requests.post(url, json = payload)
                if response.status_code == 200 : 
                    output_display.success(f"You drew the digit {response.json().get('prediction')}")
                    break

        else :
            output_display.warning("The canvas is blank.")

