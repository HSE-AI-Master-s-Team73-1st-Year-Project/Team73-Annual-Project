import os
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor
import asyncio
import streamlit as st
from .api_requests import (
    generate_images,
    load_new_adapter_checkpoint,
    change_adapter,
    change_model,
    get_available_adapter_checkpoints,
    get_available_model_types,
    get_current_model_type,
    remove,
    remove_all,
)


LOG_DIR = "/home/chaichuk/Team73-Annual-Project/streamlit_app/logs"


class AsyncHandler(logging.Handler):
    def __init__(self, handler):
        super().__init__()
        self.handler = handler
        self.executor = ThreadPoolExecutor(max_workers=1)

    def emit(self, record):
        self.executor.submit(self.handler.emit, record)


def setup_logger(name, log_file, level=logging.DEBUG):
    """Logger setup"""

    os.makedirs(LOG_DIR, exist_ok=True)

    new_logger = logging.getLogger(name)
    new_logger.setLevel(level)

    rotating_handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024, backupCount=5)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    rotating_handler.setFormatter(formatter)

    async_handler = AsyncHandler(rotating_handler)
    new_logger.addHandler(async_handler)

    return new_logger


def run_async(coroutine):
    """Run async function"""

    loop = asyncio.new_event_loop()
    return loop.run_until_complete(coroutine)


def display_image_grid(images, cols=2):
    """Display multiple images as grid"""

    rows = len(images) // cols + (len(images) % cols > 0)
    for i in range(rows):
        cols_list = st.columns(cols)
        for j in range(cols):
            index = i * cols + j
            if index < len(images):
                img = images[index]

                with cols_list[j]:
                    st.image(img, use_container_width=True)


logger = setup_logger("streamlit_app", "streamlit_app.log")

st.title(":blue[Anime Image Generation App] :sparkles:")

logger.info("Application started")

# GET список моделей
model_types_list = run_async(get_available_model_types())
model_types = list(model_types_list["models"].keys())
model_descriptions = list(model_types_list["models"].values())

current_model = run_async(get_current_model_type())
st.session_state.current_model = current_model["model_type"]


def format_func(x):
    """Format for model selection radio buttons"""
    if x == "anime":
        return f":rainbow[{x.upper()}] :ideograph_advantage:"
    return f"**{x.upper()}**"


selected_model = st.radio(
    "Select Model Type:", model_types, format_func=format_func, index=model_types.index(st.session_state.current_model)
)

st.caption(f"**{model_types_list["models"][selected_model]}**")

# Проверка на текущую модель
if selected_model != st.session_state.current_model:
    logger.info("Changing model type to %s.", selected_model)
    if st.button("Change StableDiffusion Version"):
        with st.spinner("Changing model version..."):
            result = run_async(change_model(selected_model))
        logger.info("Model type successfully changed to %s.", selected_model)
        st.success(result["message"])
else:
    st.success(f"The {selected_model} model is already selected.")
    logger.info("The %s model is already selected.", selected_model)

# Загрузка и генерация изображений
st.subheader("Generate Images")

st.session_state.expander_open = True

with st.expander("Input Images and Parameters", expanded=st.session_state.expander_open):
    uploaded_files = st.file_uploader("Upload Image Files", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    prompts = st.text_area("Enter prompts through new line").split("\n")
    st.caption("Optional, one for all the pictures, or the same number as the pictures")

    negative_prompts = st.text_area("Enter negative prompts through new line").split("\n")
    st.caption("Optional, one for all the pictures, or the same number as the pictures")

    params = {}
    params["scale"] = st.slider("**Scale**", min_value=0.1, max_value=1.0, value=0.6, step=0.1)

    st.caption("The degree of influence of uploaded images on generation")
    params["num_samples"] = st.slider("**Num samples**", min_value=1, max_value=6, value=4, step=1)

    st.caption("The number of images that will be generated for each uploaded image.")
    random_seed = st.number_input("Random seed (optional)", value=None)
    if random_seed is not None:
        params["random_seed"] = random_seed

    params["guidance_scale"] = st.slider("Guidance scale", min_value=0.0, max_value=20.0, value=7.5, step=0.1)
    st.caption("The degree of influence of prompts on generation process")

    params["height"] = st.number_input("Height", min_value=64, max_value=1024, value=512, step=64)
    params["width"] = st.number_input("Width", min_value=64, max_value=1024, value=512, step=64)

    params["num_inference_steps"] = st.number_input("Number of inference steps", min_value=1, value=50)

    params["device"] = st.selectbox("Device", options=["cuda", "cpu"], index=0)
    if params["device"] == "cpu":
        st.markdown(":red-background[Using cpu is extremely costly. Either reduce the parameters or change to cuda.]")

    logger.debug("Set generation params: %s.", params)

    if st.button("Start Generation", type="primary", use_container_width=True):
        if uploaded_files:
            with st.spinner("Generating images..."):
                logger.info("Starting generation.")
                logger.debug("Starting generation with params: %s.", params)
                generation_result = run_async(generate_images(uploaded_files, params, prompts, negative_prompts))
                if generation_result["code"] == 200:
                    logger.info("Generation successfull.")
                    generated_images = generation_result["result"]
                    display_image_grid(generated_images, cols=params["num_samples"])
                else:
                    st.error(generation_result["result"])
                    logger.error("Generation ERROR: %s", generation_result["result"])
        else:
            logger.info("No images uploaded!")
            st.warning("Please upload at least one image.")

# Список адаптеров
logger.info("Getting adapters list.")
adapter_list = run_async(get_available_adapter_checkpoints())

# Кнопка замены адаптера
st.subheader("Change Active IP-Adapter")
if adapter_list["models"]:
    selected_adapter = st.selectbox("Select IP-Adapter to Use", options=list(adapter_list["models"].keys()))
    st.caption(
        f"**:blue-background[{selected_adapter}] Checkpoint Description:** {adapter_list["models"][selected_adapter]}"
    )
    if st.button("Change IP-Adapter"):
        logger.info("Changing adapter to %s", selected_adapter)
        result = run_async(change_adapter(selected_adapter))
        logger.info("Adapter changed to: %s", selected_adapter)
        st.success(f"{result['message']}. Please reload the page.")
else:
    logger.warning("No adapters available to change!")
    st.warning("No adapters available to change.")

# Загрузка нового чекпоинта адаптера
st.subheader("Upload New IP_Adapter Checkpoint")
new_adapter_file = st.file_uploader("Upload Adapter Checkpoint (.bin)", type=["bin"])
new_adapter_id = st.text_input("New Adapter ID")
new_adapter_description = st.text_input("New Adapter Description (optional)")

# Кнопка загрузки чекпоинта
if st.button("Upload Checkpoint"):
    if new_adapter_file and new_adapter_id:
        logger.info('Loading new adapter checkpoint.')
        result = run_async(load_new_adapter_checkpoint(new_adapter_file, new_adapter_id, new_adapter_description))
        logger.info("Loaded adapter with id: %s.", new_adapter_id)
        st.success(result["message"])
    else:
        logger.error("Adapter file or id not provided.")
        st.error("Please provide both a file and an ID for the new adapter.")

# Кнопка удаления адаптера
st.subheader("Remove Uploaded IP-Adapter Checkpoint")
if adapter_list["models"]:
    adapter_to_remove = st.selectbox("Select Checkpoint to Remove", options=list(adapter_list["models"].keys()))
    if st.button("Remove Selected Adapter", type="primary"):
        logger.info("Attempting to remove checkpoint %s.", adapter_to_remove)
        result = run_async(remove(adapter_to_remove))
        logger.info("Adapter %s removed.", adapter_to_remove)
        st.success(result["message"])
else:
    logger.warning("No adapters available to remove!")
    st.warning("No adapters available to remove.")

# Кнопка удаления всех адаптеров
st.subheader("Remove All Upoaded IP-Adapter Checkpoints")
if st.button("Remove All", type="primary"):
    logger.info("Removing all adapters.")
    result = run_async(remove_all())
    logger.info("All uploaded checkpoints removed!")
    st.success("All uploaded checkpoints removed")
