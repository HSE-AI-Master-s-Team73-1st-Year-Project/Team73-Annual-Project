import requests
import streamlit as st
from PIL import Image
import io


def fetch_adapter_ids():
    response = requests.get("http://localhost:8000/get_adapters_list")
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to fetch adapter IDs.")
        return []


st.title("Anime profile picture generation web app")

# Loading images
uploaded_images = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

images = []
if uploaded_images:
    for uploaded_image in uploaded_images:
        image_data = uploaded_image.read()
        image = Image.open(io.BytesIO(image_data))
        images.append(image)
        st.image(image, caption=uploaded_image.name)

# GET responses for CUDA availability, list of models and list of adapters
cuda_response = requests.get("http://localhost:8000/get_config")
cuda_available = cuda_response.json().get("device", False)

model_response = requests.get("http://localhost:8000/get_models_list")
model_name = cuda_response.json().get("model_name")

adapter_list = fetch_adapter_ids()

# Radio for choosing the model (standard or anime trained sd1.5)
model_select = st.radio("Select model", ("standard", "anime"))
requests.post("http://localhost:8000/change_model", data=model_select)

# Loading custom adapter checkpoint
uploaded_adapter = st.file_uploader("Upload ip adapter (.bin)", type=["bin"])
id_input = st.text_input("Enter adapter ID")
description_input = st.text_area("Enter description (optional)")
adapter_list.append(id_input)

if st.button("Load checkpoint"):
    if uploaded_adapter and id_input:
        files = {"file": (uploaded_adapter.name, uploaded_adapter, uploaded_adapter.type)}
        data = {
            "id": id_input,
            "description": description_input
        }

        response = requests.post("http://localhost:8000/load_new_adapter_checkpoint", data=data, files=files)

        if response.status_code == 200:
            st.success(response.json()["message"])
        else:
            st.error(f"Error: {response.text}")
    else:
        st.warning("Please fill in all required fields.")

# Selectbox for choosing an adapter based on its ID
if adapter_list:
    selected_adapter_id = st.selectbox(
        "Select Adapter ID",
        [(adapter["id"], f"{adapter['id']} - {adapter['description']}") for adapter in adapter_list]
    )

    if st.button("Change Adapter"):
        if selected_adapter_id:
            response = requests.post("http://localhost:8000/change_adapter", data={"id": selected_adapter_id})

            if response.status_code == 200:
                st.success(response.json()["message"])
            else:
                st.error(f"Error: {response.text}")
        else:
            st.warning("Please select an adapter ID.")
else:
    st.warning("No adapters available.")

selected_adapter_id = st.selectbox("Select Adapter ID", adapter_list)

if st.button("Change Adapter"):
    if selected_adapter_id:
        # Make a POST request to the FastAPI endpoint
        response = requests.post("http://localhost:8000/change_adapter", data={"id": selected_adapter_id})

        # Handle the response from FastAPI
        if response.status_code == 200:
            st.success(response.json()["message"])
        else:
            st.error(f"Error: {response.text}")
    else:
        st.warning("Please select an adapter ID.")

adapter = st.file_uploader('Choose adapter', type=['pth'])
requests.post("http://localhost:8000/change_adapter", data=adapter)

if cuda_available:
    comp_method = st.radio("Select computation method:", ('cpu', 'cuda'))
else:
    comp_method = 'cpu'

st.write(f'Using: {comp_method} for computations.')

if st.button('Begin anime conversion'):
    for image in images:
        if image is not None:
            result = requests.post("http://localhost:8000/generate_image", data=[image, comp_method])
            image_path = result.json()
            gen_image = Image.open(image_path.get('name'))
            st.image(image, width=500)

