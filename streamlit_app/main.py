import requests
import streamlit as st
from PIL import Image
import io


st.title('Anime profile picture generation web app')

uploaded_images = st.file_uploader('Upload Images', type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

model_name = st.selectbox('Choose model method', ['sd_1.5', 'anime_sd_1.5'])
requests.post("http://localhost:8000/change_model", data=model_name)

adapter = st.file_uploader('Choose adapter', type=['pth'])
requests.post("http://localhost:8000/change_adapter", data=adapter)

cuda_response = requests.get("http://localhost:8000/check_cuda")
cuda_available = cuda_response.json().get('cuda_available', False)

if cuda_available:
    comp_method = st.radio("Select computation method:", ('cpu', 'cuda'))
else:
    comp_method = 'cpu'

st.write(f'Using: {comp_method} for computations.')

images = []
if uploaded_images:
    for uploaded_image in uploaded_images:
        image_data = uploaded_image.read()
        image = Image.open(io.BytesIO(image_data))
        images.append(image)
        st.image(image, caption=uploaded_image.name)

if st.button('Begin anime conversion'):
    for image in images:
        if image is not None:
            result = requests.post("http://localhost:8000/generate_image", data=[image, comp_method])
            image_path = result.json()
            gen_image = Image.open(image_path.get('name'))
            st.image(image, width=500)

