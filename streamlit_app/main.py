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


# Сетап логгера для записи в logs/ (возможно переместить позже в отдельный файл)
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

# GET список моделей
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
        logger.info(f"Model successfully changed to {selected_model}")
        st.success(result)
else:
    logger.info(f"The {selected_model} model is already selected.")

# GET список адаптеров
adapter_list = run_async(get_adapters_list())

# Проверка на адаптеры
if adapter_list:
    st.subheader("Available Adapters")
    for adapter in adapter_list:
        st.write(f"ID: {adapter['id']}, Description: {adapter.get('description', None)}")
else:
    logger.info("No adapters available!")
    st.warning("No adapters available, please upload at least one.")

# Загрузка нового чекпоинта адаптера
st.subheader("Load New Adapter")
new_adapter_file = st.file_uploader("Upload Adapter Checkpoint (.bin)", type=["bin"])
new_adapter_id = st.text_input("New Adapter ID")
new_adapter_description = st.text_input("New Adapter Description (optional)")

# Кнопка загрузки чекпоинта
if st.button("Load New Adapter"):
    if new_adapter_file and new_adapter_id:
        file_bytes = io.BytesIO(new_adapter_file.read())
        result = run_async(load_new_adapter_checkpoint(file_bytes, new_adapter_id, new_adapter_description))
        logger.info(f"Loaded adapter with id: {new_adapter_id}")
        st.success(f"Adapter loaded: {result['message']}")
    else:
        logger.info("")
        st.error("Please provide both a file and an ID for the new adapter.")

# Кнопка замены адаптера
st.subheader("Change Active Adapter")
if adapter_list:
    selected_adapter = st.selectbox("Select Adapter to Use", options=[adapter['id'] for adapter in adapter_list])
    if st.button("Change Adapter"):
        result = run_async(change_adapter(selected_adapter))
        logger.info(f"Adapter changed to: {selected_adapter}")
        st.success(f"Adapter changed: {result['message']}")
else:
    logger.info("No adapters available to change!")
    st.warning("No adapters available to change.")

# Кнопка удаления адаптера
st.subheader("Remove Adapter")
if adapter_list:
    adapter_to_remove = st.selectbox("Select Adapter to Remove", options=[adapter['id'] for adapter in adapter_list])
    if st.button("Remove Selected Adapter"):
        result = run_async(remove(adapter_to_remove))
        logger.info(f"Adapter removed: {adapter_to_remove}")
        st.success(f"Adapter removed: {result['message']}")
else:
    logger.info("No adapters available to remove!")
    st.warning("No adapters available to remove.")

# Кнопка удаления всех адаптеров
st.subheader("Remove All Adapters")
if st.button("Remove All Adapters"):
    result = run_async(remove_all())
    logger.info("All adapters removed!")
    st.success(f"All adapters removed: {result['message']}")

# Кнопка загрузки и генерации изображений
with st.expander("Input Images and Parameters"):
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
            logger.info("No images uploaded!")
            st.warning("Please upload at least one image.")

