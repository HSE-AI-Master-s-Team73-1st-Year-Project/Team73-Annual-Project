import requests
import aiohttp
import asyncio
import streamlit as st
import io
import os
import logging
from logging.handlers import RotatingFileHandler
from PIL import Image


def setup_logger(name, log_file, level=logging.INFO):
    os.makedirs("logs", exist_ok=True)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    handler = RotatingFileHandler(f"logs/{log_file}", maxBytes=1024 * 1024, backupCount=5)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def get_models_list():
    response = requests.get("http://localhost:8000/get_models_list")
    return response.json()


def get_adapter_list():
    response = requests.get("http://localhost:8000/get_adapters_list")
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to fetch adapter IDs.")
        return []


def change_model(new_model):
    response = requests.post(
        "http://localhost:8000/change_model",
        json={"model_type": new_model}
    )
    if response.status_code == 200:
        st.session_state.current_model = new_model
        return response.json()["message"]
    else:
        return f"Error: {response.json()["detail"]}"


def change_adapter(new_adapter):
    response = requests.post(
        "http://localhost:8000/change_adapter",
        json={"id": new_adapter}
    )
    if response.status_code == 200:
        st.session_state.current_model = new_adapter
        return response.json()["message"]
    else:
        return f"Error: {response.json()["detail"]}"


# Setting up logger to print out logs in "logs/" folder
logger = setup_logger("streamlit_app", "app.log")

st.title("Anime profile picture generation web app")

logger.info("Application started")
logger.debug("This is a debug message")
logger.warning("This is a warning")

# Loading images
uploaded_images = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# images = []
# if uploaded_images:
#     for uploaded_image in uploaded_images:
#         image_data = uploaded_image.read()
#         image = Image.open(io.BytesIO(image_data))
#         images.append(image)
#         st.image(image, caption=uploaded_image.name)

# GET responses for CUDA availability, list of models and list of adapters
# cuda_response = requests.get("http://localhost:8000/get_config")
# cuda_available = cuda_response.json().get("device", False)

# model_response = requests.get("http://localhost:8000/get_models_list")
# model_name = model_response.json().get("id")

model_list = get_models_list()
st.session_state.current_model = model_list.json().get("id")[0]

selected_model = st.radio(
    "Select Model Type:",
    ("standard", "anime"),
    index=0 if st.session_state.current_model == "anime" else 1
)

if selected_model != st.session_state.current_model:
    if st.button("Change Model"):
        with st.spinner("Changing model..."):
            result = change_model(selected_model)
        st.success(result)
        logger.info(f"Model successfully changed to {selected_model}")
else:
    logger.info(f"The {selected_model} model is already selected.")

adapter_list = get_adapter_list()
st.session_state.current_adapter = model_list.json().get("id")[0]

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

comp_method = st.radio("Select computation method:", ("cpu", "cuda"))
if comp_method == "cpu":
    st.write("Using cpu for computations may be costly. Either reduce the parameters or change to cuda.")

# if st.button('Begin anime conversion'):
#     for image in images:
#         if image is not None:
#             result = requests.post("http://localhost:8000/generate_image", data=[image, comp_method])
#             image_path = result.json()
#             gen_image = Image.open(image_path.get('name'))
#             st.image(image, width=500)
