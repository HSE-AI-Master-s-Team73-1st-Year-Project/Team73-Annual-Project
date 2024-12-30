import requests
import aiohttp
import asyncio
import streamlit as st
import os
import logging
from api_requests import *
from logging.handlers import RotatingFileHandler


# Запускаем асинх код в другом потоке
def run_async(coroutine):
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(coroutine)


# Сетап логгера для записи в logs/
def setup_logger(name, log_file, level=logging.INFO):
    os.makedirs("logs", exist_ok=True)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    handler = RotatingFileHandler(f"logs/{log_file}", maxBytes=1024*1024, backupCount=5)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


logger = setup_logger("streamlit_app", "app.log")

st.title("Anime profile picture generation web app")

logger.info("Application started")
logger.debug("This is a debug message")
logger.warning("This is a warning")

model_list = run_async(get_models_list())
st.session_state.current_model = model_list.json().get("id")[0] # присваивает аниме вариант по стандарту

# Радио кнопки выбора модели
selected_model = st.radio(
    "Select Model Type:",
    ("standard", "anime"),
    index=0 if st.session_state.current_model == "anime" else 1
)

# Проверка на текущую модель
if selected_model != st.session_state.current_model:
    if st.button("Change Model"):
        with st.spinner("Changing model..."):
            result = change_model(selected_model)
        st.success(result)
        logger.info(f"Model successfully changed to {selected_model}")
else:
    logger.info(f"The {selected_model} model is already selected.")

adapter_list = get_adapters_list()
st.session_state.current_adapter = adapter_list.json().get("id")[0]

# Загрузка нового чекпоинта адаптера
uploaded_adapter = st.file_uploader("Upload ip adapter (.bin)", type=["bin"])
id_input = st.text_input("Enter adapter ID")
description_input = st.text_area("Enter description (optional)")
adapter_list.append(id_input)

# Кнопка загрузки чекпоинта
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
            logger.warning(f"Error: {response.text}")
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

with st.expander("Input images and parameters"):
    uploaded_files = st.file_uploader("Upload Image Files", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    params = {}
    params['prompts'] = st.text_area("Enter prompts through new line (optional)").split('\n')
    params['negative_prompts'] = st.text_area("Enter negative prompts through new line (optional)").split('\n')
    params['scale'] = st.slider("Scale", min_value=0.1, max_value=1.0, value=0.6, step=0.1)
    params['num_samples'] = st.number_input("Number of samples", min_value=1, value=1)
    params['random_seed'] = st.number_input("Random seed", value=42)
    params['guidance_scale'] = st.slider("Guidance scale", min_value=0.0, max_value=20.0, value=7.5, step=0.1)
    params['height'] = st.number_input("Height", min_value=64, max_value=1024, value=512, step=64)
    params['width'] = st.number_input("Width", min_value=64, max_value=1024, value=512, step=64)
    params['num_inference_steps'] = st.number_input("Number of inference steps", min_value=1, value=50)
    params['device'] = st.selectbox("Device", options=["cuda", "cpu"], index=0)
    if params['device'] == "cpu":
        st.write("Using cpu for computations may be costly. Either reduce the parameters or change to cuda.")

    if st.button("Generate Images"):
        if uploaded_files:
            with st.spinner("Generating images..."):
                generated_images = run_async(generate_images(uploaded_files, params))
            for i, img in enumerate(generated_images):
                st.image(img, caption=f"Generated Image {i + 1}", use_column_width=True)
        else:
            logger.warning("No images uploaded!")
            st.warning("Please upload at least one image.")

