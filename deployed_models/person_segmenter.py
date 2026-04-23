# get packages
import streamlit as st
import numpy as np
import io
import base64
import json
from PIL import Image
import requests
from torchvision.utils import draw_segmentation_masks
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import v2

# lambda endpoint url
url = "https://nc2xhheqaivoywho43qhnvzwqm0agtzo.lambda-url.us-east-1.on.aws/"

def generate_mask(uploaded_image) : 

    # image
    image_tensor_for_display = F.resize(F.to_tensor(uploaded_image), size = (512, 512), antialias = True)
    uploaded_image = image_tensor_for_display
    uploaded_image = F.to_pil_image(uploaded_image)

    # convert to base64 to send. PNG is lossless. JPEG is not.
    buffered = io.BytesIO()
    uploaded_image.save(buffered, format = "PNG")
    uploaded_image_b64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')

    payload = {
            "data": uploaded_image_b64_string
        }

    try :
        response = requests.post(url, json = payload)
        mask_b64 = response.json().get("mask")
        mask_bytes = base64.b64decode(mask_b64)
        mask_image = Image.open(io.BytesIO(mask_bytes))

        # segment image
        img_uint8 = (image_tensor_for_display * 255).to(torch.uint8)
        bool_mask = F.to_tensor(mask_image).bool()
        segmented_image = F.to_pil_image(draw_segmentation_masks(img_uint8, bool_mask, alpha = 0.40, colors = "blue"))


        # apply elastic
        elastic_transformer = v2.ElasticTransform(alpha = 1000.0)
        watery_image = elastic_transformer(image_tensor_for_display)
        watery_uint8 = (watery_image * 255).to(torch.uint8)
        watery = torch.where(bool_mask, img_uint8, watery_uint8)
        watery_image = F.to_pil_image(watery)

        return segmented_image, watery_image
    
    except :
        return uploaded_image, uploaded_image



# configure the application layout
st.set_page_config(page_title = "Person Segmenter and Background Filter", layout = "centered")
st.title("Person Segmenter and Background Filter")
st.write("Upload an image and press segment. If people are present in the image they will be semantically segmented using a UNet with a ResNet18 backbone" \
" and the background will be filtered.")
column1, column2, column3, column4 = st.columns((1,1,1, 1))
uploaded_file = None

with column1 :
    st.write("### Control Panel")
    uploaded_file = st.file_uploader("Upload an image.")
    infer_button = st.button("Segment", use_container_width = True)

    
image = None
segmented_image = None
watery_image = None
if uploaded_file is not None :
    image = Image.open(uploaded_file)

with column2 :
    st.divider()
    st.write("Unaltered Image")
    if image :
        st.image(image, use_container_width= True)

if infer_button and image :
    with st.spinner("Processing...") :
        segmented_image, watery_image = generate_mask(image)


with column3:
    st.divider() 
    st.write("Segmented Image")
    if segmented_image :
        st.image(segmented_image, use_container_width = True)

with column4:
    st.divider() 
    st.write("Filtered Image")
    if watery_image:
        st.image(watery_image, use_container_width = True)



